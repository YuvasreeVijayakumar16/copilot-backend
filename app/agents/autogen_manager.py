import logging
import os
import time
import requests
import json as _json
from typing import List, Dict, Any, Optional, Callable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from app.services.agent_servies import load_agent_config, GET_ALL_AGENTS_URL
from app.agents.agent_conversation import AgentNode, MessageBus, _client
from app.utils.schema_reader import get_schema_and_sample_data
from app.db.sql_connection import execute_sql_query
from app.utils.ppt_generator import generate_direct_response
import pandas as pd

logger = logging.getLogger(__name__)

class AgentManager:
    SIMPLE_TASK_KEYWORDS = ["simple", "quick", "basic", "single", "only"]
    _agents_cache: Optional[List[AgentNode]] = None
    _agents_cache_time: Optional[float] = None
    _agents_cache_ttl: int = 300

    def __init__(self):
        self.bus = MessageBus()
        # Scalability: thread pool allowed for limited parallelism; env var controls size
        self._executor = ThreadPoolExecutor(max_workers=int(os.getenv("AGENT_MAX_WORKERS", "4")))

    def _is_simple_task(self, task: str) -> bool:
        return any(keyword in task.lower() for keyword in self.SIMPLE_TASK_KEYWORDS)

    def _execute_with_retries(self, agent: AgentNode, task: str, context: Optional[Dict[str, Any]] = None, retries: int = 2) -> Dict[str, Any]:
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                logger.info("Attempting agent execution", extra={"agent": agent.name, "attempt": attempt})
                res = agent.execute(task, context)
                if isinstance(res, dict) and res.get("error"):
                    last_exc = res.get("error")
                    # Robustness: agent returned an error; record and retry according to policy.
                    logger.warning("Agent returned error", extra={"agent": agent.name, "error": res.get("error")})
                    continue
                return res
            except Exception as e:
                last_exc = str(e)
                # Fault tolerance: log and apply a small backoff before retrying
                logger.exception("Execution exception", extra={"agent": agent.name, "attempt": attempt})
                time.sleep(0.5 * attempt)

        return {"error": f"execution_failed_after_retries: {last_exc}"}

    def _agents_from_configs(self, configs: List[Dict[str, Any]]) -> List[AgentNode]:
        nodes = []
        for cfg in configs:
            name = cfg.get("name") or cfg.get("Name")
            purpose = cfg.get("purpose") or cfg.get("Purpose") or ""
            # Modularity: wrap raw config into AgentNode instances
            nodes.append(AgentNode(name=name, purpose=purpose, config=cfg))
        return nodes

    def discover_all_agents(self) -> List[AgentNode]:
        now = time.time()
        if (
            self._agents_cache is not None
            and self._agents_cache_time is not None
            and (now - self._agents_cache_time) < self._agents_cache_ttl
        ):
            logger.info("Returning cached agent node list")
            return self._agents_cache

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to fetch all agents from API: {GET_ALL_AGENTS_URL} (attempt {attempt + 1}/{max_retries})")
                    resp = requests.get(GET_ALL_AGENTS_URL, timeout=30)
                    resp.raise_for_status()
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        sleep_time = 5 * (attempt + 1)
                        logger.warning(f"Timeout fetching agents, retrying in {sleep_time} seconds", extra={"attempt": attempt + 1})
                        time.sleep(sleep_time)
                    else:
                        raise  # Re-raise after last attempt

            table = resp.json().get("Table", [])
            if not table:
                logger.warning("API returned no agent records in the 'Table' field.")
                return []

            names = sorted(list({row.get("Name") for row in table if row.get("Name" )}))
            configs = []
            for name in names:
                cfg = load_agent_config(name)
                if cfg:
                    configs.append(cfg.dict())
                else:
                    logger.warning(f"Failed to load config for agent: {name}")

            # Convert configs to AgentNode (encapsulation + prepare for health/auth checks)
            nodes = self._agents_from_configs(configs)
            self._agents_cache = nodes
            self._agents_cache_time = now
            logger.info(f"Discovered {len(nodes)} agent nodes")
            return nodes

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching agent list from API: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error("Error discovering all agents", exc_info=True, extra={"error": str(e)})
            return []

    def discover_agents(self, names: List[str]) -> List[AgentNode]:
        all_agents = self.discover_all_agents()
        if not names:
            return all_agents
        lower = [n.strip().lower() for n in names]
        return [a for a in all_agents if a.name and a.name.strip().lower() in lower]

    def route(self, task: str, agents: List[AgentNode]) -> Optional[AgentNode]:
        if not agents:
            logger.warning("Routing failed: No agents provided to router")
            return None

    # Coordination & Cooperation: heuristic routing tries to find an agent whose
    # purpose words overlap with the task. This is a lightweight task allocation
    # strategy. If ambiguous, fallback to LLM routing (adaptability + goal-oriented).
        task_lower = task.lower()
        best = None
        best_score = 0
        for a in agents:
            score = 0
            if a.purpose:
                for token in a.purpose.lower().split():
                    if token in task_lower:
                        score += 1
            # small bias if agent is healthy
            # Robustness: prefer healthy agents slightly
            if a.health_check():
                score += 0.1
            if score > best_score:
                best_score = score
                best = a

        if best:
            logger.info("Heuristic routing selected agent", extra={"agent": best.name, "score": best_score})
            return best

        # Fallback to LLM-based routing for ambiguous cases
        try:
            agent_descriptions = "\n".join([f"- {a.name}: {a.purpose}" for a in agents])
            prompt = (
                f"You are a routing assistant. Choose the best agent by name from the list below for the task. Respond with only the agent name.\n\n"
                f"Agents:\n{agent_descriptions}\n\nTask: {task}"
            )
            # Adaptability: use LLM to help select an agent for ambiguous tasks
            resp = _client.chat.completions.create(
                model=os.getenv("AUTOGEN_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            chosen_name = (resp.choices[0].message.content or "").strip().replace('"', '')
            for a in agents:
                if a.name and a.name.strip().lower() == chosen_name.lower():
                    logger.info("LLM routing selected agent", extra={"agent": a.name})
                    return a
            logger.warning("LLM provided agent not in list, falling back to first agent")
        except Exception:
            logger.exception("LLM routing failed")

        return agents[0]

    def _run_single(self, task: str, agent: AgentNode, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Decouple actual execution via AgentNode and provide retries
        return self._execute_with_retries(agent, task, context=context, retries=int(os.getenv("AGENT_RETRIES", "2")))

    def _maybe_evaluate(self, result: Dict[str, Any], criteria: Optional[str]) -> bool:
        if not criteria:
            return True
        # Placeholder for evaluation logic; always True for now
        return True

    def _combine_results_with_llm(self, task: str, steps: List[Dict[str, Any]]) -> str:
        """Use LLM to combine and summarize results from all agent steps."""

        if not steps:
            return "No results to combine."

        # Prepare a summary of steps for the LLM
        step_summaries = []
        for i, step in enumerate(steps, 1):
            agent = step.get("agent", "Unknown")
            task_desc = step.get("task", "")
            result = step.get("result", {})
            if "error" in result:
                summary = f"Step {i} ({agent}): Failed - {result['error']}"
            else:
                answer = result.get("answer", "No answer")
                preview_rows = result.get("preview_rows", [])
                file_path = result.get("file_path", None) 
                file_type = result.get("file_type", None)
                upload_status = result.get("file_type", None)

                preview_str = ""
                if preview_rows:
                    preview_str = "Preview Rows:\n" + "\n".join([f"x:{idx+1} {', '.join([f'{k}:{v}' for k,v in row.items()])}" for idx, row in enumerate(preview_rows)])

                file_info = ""
                if file_path and file_type:
                    file_info = f"File generated: {file_path} (type: {file_type})\n"
                if upload_status:
                    file_info += f"Upload status: {upload_status}\n"

                summary = f"Step {i} ({agent}): {answer[:500]}...\n{preview_str}{file_info}"
            step_summaries.append(summary)

        steps_text = "\n".join(step_summaries)

        prompt = (
            f"You are an assistant that combines and summarizes results from multiple agents working on a task.\n\n"
            f"Original Task: {task}\n\n"
            f"Agent Results:\n{steps_text}\n\n"
            f"Provide a concise, coherent summary of the combined results. If there are errors, note them. "
            f"If a file was generated, mention it. Keep the summary professional and focused."
        )

        try:
            resp = _client.chat.completions.create(
                model=os.getenv("AUTOGEN_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            combined = (resp.choices[0].message.content or "").strip()
            logger.info("Combined results generated by LLM")
            return combined
        except Exception as e:
            logger.exception("Failed to combine results with LLM")
            return "Failed to generate combined summary."

    def run_workflow(self, plan: List[Dict[str, Any]], candidate_agents: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Starting workflow execution")
        agent_list = candidate_agents if candidate_agents else None
        agents = self.discover_agents(agent_list) if agent_list else self.discover_all_agents()

        if not agents:
            logger.error("Aborting workflow: No agents were discovered or loaded.")
            return {"error": "No agents available for workflow execution", "steps": []}

        logger.info(f"Discovered {len(agents)} agents for workflow execution: {[a.name for a in agents]}")

        # Use provided initial context if present so callers can seed values like
        # output_format, created_by, and encrypted_filename that should be
        # forwarded into agent execution and ultimately into the orchestrator.
        ctx: Dict[str, Any] = context or {}

        results: List[Dict[str, Any]] = []

        for idx, step in enumerate(plan, start=1):
            task = step.get("task", "").strip()
            if not task:
                continue

            original_task = task
            try:
                if ctx:
                    task = task.format(**ctx)
                logger.info(f"Executing step {idx}: '{task}'")
            except KeyError as e:
                logger.error(f"Formatting failed, missing key: {e}")
                results.append({"agent": "N/A", "task": original_task, "result": {"error": f"Formatting failed, missing context key: {e}"}})
                continue
            except Exception:
                logger.warning("Could not format task; using original string")

            # Coordination: route each step to a chosen agent and execute
            agent = self.route(task, agents)
            if not agent:
                results.append({"agent": "N/A", "task": task, "result": {"error": "Routing failed"}})
                continue

            # If the step can be done in parallel and marked as such, support parallel execution (simple pattern)
            try:
                r = self._run_single(task, agent, ctx)
            except Exception as e:
                logger.exception("Step execution failed", extra={"step": idx, "agent": agent.name})
                r = {"error": str(e)}

            step_result = {"agent": agent.name, "task": task, "result": r}
            results.append(step_result)

            # publish the result for interested subscribers (Communication Protocols)
            try:
                self.bus.publish(f"agent.{agent.name}.result", step_result)
            except Exception:
                logger.exception("Failed to publish result to bus")

            # Environment Awareness & Goal-Oriented Behavior:
            # update context with step outputs to feed subsequent steps
            res = step_result.get("result") or {}
            if "error" not in res and res.get("answer"):
                output_key = step.get("output_key", f"answer_{idx}")
                ctx[output_key] = res.get("answer")
                logger.info(f"Context updated: {output_key}")

        return {"steps": results}

    def plan_from_task(self, task: str) -> List[Dict[str, Any]]:
        default = [{"task": task.strip(), "output_key": "answer"}]
        try:
            model = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
            logger.info("Generating task plan from GPT", extra={"task": task, "model": model})
            prompt = (
                "You are an intelligent assistant that creates a JSON plan to solve a user's task. "
                "Break the main task into 2-4 sequential steps.\n\n"
                "RULES:\n"
                "- Return a valid JSON array of objects. Do not add comments or any other text.\n"
                "- Each object must have a 'task' and an 'output_key'.\n"
                "- The 'task' for a later step MUST use the 'output_key' from a previous step as a placeholder in curly braces if it needs that data. For example: 'Analyze the sales data for {product_name}'.\n"
                "- The first task should address the first logical part of the original user's question.\n"
                "- Ensure 'output_key' is a simple, valid variable name.\n\n"
                "Now, generate the plan for this task:\n"
                f"Task: {task}"
            )
            # Adaptability & Goal-Oriented Behavior: use LLM to decompose the user's
            # task into a small sequential plan with explicit output_keys to enable
            # goal-directed multi-step execution.
            resp = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            txt = (resp.choices[0].message.content or "").strip()
            logger.info("Raw plan response from GPT", extra={"response": txt})
            if not txt:
                logger.warning("Empty plan returned by GPT. Falling back to default plan.")
                return default
            plan = _json.loads(txt)
            if isinstance(plan, list) and all(isinstance(step, dict) for step in plan):
                logger.info("Parsed valid plan from GPT", extra={"steps": len(plan)})
                return plan
            else:
                logger.warning("Invalid plan format returned by GPT", extra={"raw_response": txt})
                return default
        except Exception as e:
            logger.error("Failed to generate plan from task", exc_info=True, extra={"error": str(e), "task": task})
            return default

    def _fetch_data_for_task(self, task: str) -> pd.DataFrame:
        """Fetch relevant data for the task using SQL query generation."""
        try:
            structured_schema, schema_text, sample_data = get_schema_and_sample_data()
            logger.info("Available tables", extra={"tables": list(structured_schema.keys())})
            if not structured_schema:
                logger.warning("No schema available for data fetching")
                return pd.DataFrame()

            # Use LLM to generate SQL query based on task
            prompt = (
                f"Generate a SQL query to fetch all data from the most relevant table for the task: '{task}'\n\n"
                f"Database schema (table(columns)):\n{schema_text}\n\n"
                "Important: Use the exact table and column names from the schema. Use SELECT TOP 100 * FROM table_name to fetch all columns. Return only the SQL query, no explanations or markdown."
            )
            resp = _client.chat.completions.create(
                model=os.getenv("AUTOGEN_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            sql_query = (resp.choices[0].message.content or "").strip()
            # Clean up the query
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            logger.info("Generated SQL query", extra={"query": sql_query})

            if not sql_query:
                return pd.DataFrame()

            # Execute the query
            result = execute_sql_query(sql_query)
            if isinstance(result, dict) and "error" in result:
                logger.error("SQL query failed", extra={"error": result["error"]})
                return pd.DataFrame()

            df = pd.DataFrame(result)
            if df.empty:
                logger.warning("Query returned no data")
                return pd.DataFrame()

            df_clean = df.replace([pd.NA, float('inf'), -float('inf')], pd.NA).fillna("null")
            logger.info("Fetched data", extra={"rows": len(df_clean), "columns": list(df_clean.columns)})
            return df_clean

        except Exception as e:
            logger.exception("Failed to fetch data for task")
            return pd.DataFrame()

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample inventory data for demonstration."""
        data = {
            "Material": ["Steel", "Aluminum", "Copper", "Plastic", "Glass", "Wood", "Rubber", "Fabric"],
            "Current_Stock": [150, 200, 80, 300, 120, 90, 250, 180],
            "Reorder_Point": [100, 150, 50, 200, 80, 60, 150, 120],
            "Fill_Rate": [0.95, 0.88, 0.92, 0.97, 0.85, 0.90, 0.93, 0.89],
            "Turnover_Rate": [4.2, 3.8, 5.1, 2.9, 3.5, 4.0, 3.2, 3.7]
        }
        df = pd.DataFrame(data)
        df["Understock"] = df["Current_Stock"] < df["Reorder_Point"]
        logger.info("Created sample data", extra={"rows": len(df)})
        return df

    def plan_and_run(self, task: str, candidate_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        plan = self.plan_from_task(task)
        logger.info("Plan from GPT", extra={"plan": plan})

        # Fetch data for the task
        df_clean = self._fetch_data_for_task(task)

        # Allow callers to pass an initial context via the candidate_agents slot
        # if they provided a dict there (backwards-compatible: most callers
        # pass a list or None). If candidate_agents is a dict, treat it as
        # context. Otherwise interpret as agent whitelist.
        context = {"df_clean": df_clean}
        agents_param = candidate_agents
        if isinstance(candidate_agents, dict):
            context.update(candidate_agents)
            agents_param = None

        result = self.run_workflow(plan, agents_param, context)

        # Combine results from all agents into a summary
        steps = result.get("steps", [])
        combined_results = self._combine_results_with_llm(task, steps)

        return {
            "plan": plan,
            "steps": steps,
            "combined_results": combined_results
        }
