# agent_services.py

import os
import json
import numpy as np
import requests
from datetime import datetime
from collections import defaultdict
import logging
import re
from uuid import uuid4

from app.db.sql_connection import execute_sql_query
from app.utils.ppt_generator import generate_ppt, generate_excel, generate_word, generate_insights, generate_direct_response
from app.utils.schema_reader import get_schema_and_sample_data
from app.utils.gpt_utils import generate_sql_query
from app.utils.gpt_utils import is_question_relevant_to_purpose
from app.utils.gpt_utils import serialize
from app.utils.llm_validator import validate_purpose_and_instructions

from app.models.agent import AgentConfig

logger = logging.getLogger("app.services.agent_servies")
MAX_ROWS = 1000
REQUEST_TIMEOUT = 20
MAX_QUESTION_LENGTH = 5000
API_ROOT = "https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/"
GET_ALL_AGENTS_URL = f"{API_ROOT}GetAgentdetails"
EDIT_AGENT_URL = f"{API_ROOT}UpdateAgentDetails"
API_URL = f"{API_ROOT}GetAgentDetails"
PUBLISH_AGENT_URL = f"{API_ROOT}UpdateAgentDetails"

# --- Guardrail helpers ---
INJECTION_PATTERNS = [
    r"(?i)ignore (all|any|previous) (instructions|rules)",
    r"(?i)act as",
    r"(?i)system prompt",
    r"(?i)developer mode",
    r"(?i)jailbreak",
]

FORBIDDEN_SQL_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "merge", "grant", "revoke", "exec", "execute", "xp_"
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",               # SSN-like
    r"\b\d{13,19}\b",                       # credit card-ish
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", # Phone numbers
    r"\b\w+@\w+\.\w+\b"                     # Email (simple check)
]

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
SAFE_FILENAME_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

def validate_question_safety(question: str) -> tuple[bool, list[str]]:
    reasons = []
    if not (question and question.strip()):
        reasons.append("Empty question")
    if len(question) > MAX_QUESTION_LENGTH:
        reasons.append("Question too long")
    for pat in INJECTION_PATTERNS:
        if re.search(pat, question):
            reasons.append("Potential prompt injection detected")
            break
    for pat in PII_PATTERNS:
        if re.search(pat, question):
            reasons.append("Potential PII in question")
            break
    return (len(reasons) == 0, reasons)

def validate_created_by_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.match(email))

def validate_safe_filename(name: str) -> bool:
    return bool(name and SAFE_FILENAME_REGEX.match(name))

# --- Ethical guardrails ---
ETHICAL_CATEGORIES = {
    "hate": [r"(?i)\b(hate|exterminate|genocide)\b", r"(?i)\b(slur|racial epithet)\b"],
    "violence": [r"(?i)\b(kill|murder|assassinate|bomb)\b"],
    "self_harm": [r"(?i)\b(self\s*h(a|)rm|suicide|kill\s*myself)\b"],
    "sexual": [r"(?i)\b(explicit|porn|sexual act)\b"],
    "illegal": [r"(?i)\b(hack|ddos|credit card dump|buy drugs)\b"],
}

def validate_ethical_use(question: str) -> tuple[bool, list[str]]:
    violations = []
    if not question:
        return True, violations
    for category, patterns in ETHICAL_CATEGORIES.items():
        for pat in patterns:
            if re.search(pat, question):
                violations.append(category)
                break
    return (len(violations) == 0, violations)

def is_sql_read_only(sql: str) -> bool:
    if not sql:
        return False
    lowered = sql.lower()
    if ";" in lowered.strip().rstrip(";"):
        return False
    for kw in FORBIDDEN_SQL_KEYWORDS:
        if kw in lowered:
            return False
    return lowered.strip().startswith("select")

def enforce_sql_row_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    if not sql:
        return sql
    lowered = sql.lstrip().lower()
    if not lowered.startswith("select"):
        return sql
    # If already has TOP or OFFSET/FETCH, leave as-is
    if re.search(r"(?i)\bselect\s+top\s+\d+", sql) or re.search(r"(?i)offset\s+\d+\s+rows", sql):
        return sql
    # Insert TOP N after SELECT or SELECT DISTINCT
    return re.sub(r"(?i)^\s*select\s+(distinct\s+)?", lambda m: f"{m.group(0)}TOP {max_rows} ", sql, count=1)

def _normalize_table_name(name: str) -> str:
    # Remove brackets or quotes and split alias/commas
    name = name.strip().strip('[]').strip('`').strip('"')
    # Remove schema alias like dbo.Table
    parts = name.split()
    if parts:
        name = parts[0]
    # Remove trailing commas
    return name.strip(',')

def extract_sql_tables(sql: str) -> list[str]:
    if not sql:
        return []
    tables = []
    for pattern in [r"(?i)\bfrom\s+([\w\[\]`\.\"]+)", r"(?i)\bjoin\s+([\w\[\]`\.\"]+)"]:
        for match in re.finditer(pattern, sql):
            tables.append(_normalize_table_name(match.group(1)))
    return tables

def validate_sql_tables(sql: str, allowed_tables: list[str]) -> tuple[bool, list[str]]:
    referenced = extract_sql_tables(sql)
    if not referenced:
        return True, []
    normalized_allowed = set([_normalize_table_name(t) for t in (allowed_tables or [])])
    violations = [t for t in referenced if _normalize_table_name(t) not in normalized_allowed]
    return (len(violations) == 0, violations)

AGENT_DIR = "agents"
os.makedirs(AGENT_DIR, exist_ok=True)

ALLOWED_CAPABILITIES = [
    "Summarize results", "Generate output as text", "Generate output as PPT",
    "Generate output as Excel", "Generate output as Word", "Highlight anomalies",
    "Generate visual reports", "Provide data-driven recommendations",
    "Assist with data analysis", "Create data-driven insights", "Automate repetitive tasks",
    "Support decision-making", "Charts and graphs", "Data validation", "Data visualization"
]

def _ensure_list(data):
    """
    Ensures the input is a list of clean, stripped strings.
    Handles single comma-separated strings within lists.
    """
    if isinstance(data, list):
        data_str = data[0] if len(data) == 1 and isinstance(data[0], str) else data
    elif isinstance(data, str):
        data_str = data
    else:
        return []

    if isinstance(data_str, str):
        return [item.strip() for item in data_str.split(',') if item.strip()]
    
    return [item.strip() for item in data_str if isinstance(item, str) and item.strip()]

def is_question_supported_by_capabilities(question: str, capabilities: list) -> bool:
    capability_keywords = {
        "ppt": "Generate output as PPT",
        "pptx": "Generate output as PPT",
        "presentation": "Generate output as PPT",
        "excel": "Generate output as Excel",
        "xlsx": "Generate output as Excel",
        "word": "Generate output as Word",
        "doc": "Generate output as Word",
        "docx": "Generate output as Word",
        "chart": "Charts and graphs",
        "graph": "Charts and graphs",
        "visual": "Data visualization",
        "recommend": "Provide data-driven recommendations",
        "summarize": "Summarize results",
        "insight": "Create data-driven insights",
        "anomaly": "Highlight anomalies",
        "validate": "Data validation",
        "automate": "Automate repetitive tasks",
        "assist": "Assist with data analysis",
        "support": "Support decision-making",
        "visualize": "Data visualization",
        "data-driven": "Provide data-driven recommendations",

        "data analysis": "Assist with data analysis",
        "data insights": "Create data-driven insights"
    }

    q_lower = question.lower()
    for keyword, required_capability in capability_keywords.items():
        if keyword in q_lower:
            if required_capability not in capabilities:
                return False
    return True


def capability_supports_format(capabilities: list, fmt_key: str) -> bool:
    """Tolerant check whether a capabilities list supports a requested output format."""
    if not fmt_key or not capabilities:
        return False
    caps = [c.strip().lower() for c in _ensure_list(capabilities)]
    
    # 1. Check for the canonical capability name
    req_map = {
        "ppt": "generate output as ppt",
        "excel": "generate output as excel",
        "word": "generate output as word",
    }
    if req_map.get(fmt_key, "") in caps:
        return True
    
    # 2. Check for keywords within capabilities
    keywords = {
        "ppt": ["ppt", "pptx", "presentation"],
        "excel": ["excel", "xlsx", "spreadsheet"],
        "word": ["word", "doc", "docx", "document"],
    }
    for kw in keywords.get(fmt_key, []):
        for c in caps:
            if kw in c:
                return True
    return False

    


def detect_output_format(question: str) -> str:
    q = (question or "").lower()
    fmt = "none"
    if any(x in q for x in ["ppt", "pptx", "presentation"]):
        fmt = "ppt"
    elif any(x in q for x in ["excel", "xlsx", "spreadsheet"]):
        fmt = "excel"
    elif any(x in q for x in ["word", "doc", "docx", "document"]):
        fmt = "word"
    logger.info("detect_output_format", extra={"question_preview": q[:200], "format": fmt})
    return fmt
    


def save_agent_config(agent_config: AgentConfig):
    path = f"{AGENT_DIR}/{agent_config.name}.json"
    with open(path, "w") as f:
        json.dump(agent_config.dict(), f, indent=2)
    return {"message": "Agent config saved", "path": path, "agent": agent_config.dict()}



async def handle_agent_request(data : dict):
    logger.info("handle_agent_request: start", extra={
        "has_agent_config": bool(data.get("agent_config")),
        "question_len": len((data.get("question") or "")),
        "has_schema": bool(data.get("structured_schema")),
        "has_sample_data": bool(data.get("sample_data"))
    })
    incoming_config = data.get("agent_config")
    agent_name = incoming_config.get("name") if incoming_config else None
    question = data.get("question")
    # structured_schema = data.get("structured_schema") # <-- Use input, not overwrite
    # sample_data = data.get("sample_data")           # <-- Use input, not overwrite
    encrypted_filename = data.get("encrypted_filename")
    created_by = data.get("created_by")
    # formatdata = data.get("formatdata", {}) # not used

    if not all([question, agent_name, created_by, encrypted_filename]):
        logger.warning("Missing required fields for handle_agent_request", extra={
            "has_question": bool(question),
            "has_agent_name": bool(agent_name),
            "has_created_by": bool(created_by),
            "has_encrypted_filename": bool(encrypted_filename)
        })
        return {"error": "❌ Missing one or more required fields: 'question', 'name', 'created_by', 'encrypted_filename'"}

    # Security: validate creator and filename
    if not validate_created_by_email(created_by):
        logger.warning("Invalid created_by email", extra={"created_by": created_by})
        return {"error": "❌ Invalid 'created_by' format"}
    if not validate_safe_filename(encrypted_filename):
        logger.warning("Unsafe encrypted_filename", extra={"encrypted_filename": encrypted_filename})
        return {"error": "❌ Invalid 'encrypted_filename' value"}

    # ✅ Load existing agent config
    agent_config = load_agent_config(agent_name)
    if not agent_config:
        logger.error("Agent not found", extra={"agent_name": agent_name})
        return {"error": f"❌ Agent '{agent_name}' not found"}

    # ✅ Capability enforcement
    if not is_question_supported_by_capabilities(question, agent_config.capabilities):
        logger.info("Capability check failed", extra={
            "agent_name": agent_name,
            "question": question,
            "capabilities": agent_config.capabilities
        })
        return {
            "error": f"❌ This question requires capabilities not available in agent '{agent_name}'.",
            "allowed_capabilities": agent_config.capabilities
        }

    
    # Guardrail: question safety
    ok_question, reasons = validate_question_safety(question)
    if not ok_question:
        logger.warning("Question safety violation", extra={"reasons": reasons})
        return {"error": "❌ Question rejected by safety guardrails", "reasons": reasons}

    # Ethical guardrails
    ethical_ok, ethical_violations = validate_ethical_use(question)
    if not ethical_ok:
        logger.warning("Ethical guardrail violated", extra={"violations": ethical_violations})
        return {"error": "❌ Request violates ethical guardrails", "violations": ethical_violations}

    # GPT-based semantic check
    is_relevant = await is_question_relevant_to_purpose(question, agent_config.purpose)
    if not is_relevant:
        logger.info("Purpose relevance check failed", extra={
            "agent_name": agent_name,
            "purpose": agent_config.purpose
        })
        return {"error": f"❌ Question does not align with agent's purpose: '{agent_config.purpose}'"}


    # ✅ Load schema and data 
    structured_schema, _, sample_data = get_schema_and_sample_data()
    logger.info("Loaded schema and sample data", extra={
        "tables": list(structured_schema.keys()) if isinstance(structured_schema, dict) else None
    })

    # ✅ Generate SQL
    sql_query = generate_sql_query(question, structured_schema)
    if not is_sql_read_only(sql_query):
        logger.warning("Non read-only SQL generated; rejecting", extra={"sql": sql_query[:500]})
        return {"error": "❌ Generated SQL is not read-only and was blocked by guardrails"}
    sql_query = enforce_sql_row_limit(sql_query)
 
     # Security: validate referenced tables against schema allowlist
    allowed_tables = list(structured_schema.keys()) if isinstance(structured_schema, dict) else []
    ok_tables, bad_tables = validate_sql_tables(sql_query, allowed_tables)
    if not ok_tables:
        logger.warning("SQL references unauthorized tables", extra={"bad_tables": bad_tables})
        return {"error": "❌ SQL references unauthorized tables", "tables": bad_tables}
 
    logger.info("Generated SQL query", extra={"query_len": len(sql_query or "")})
    result = execute_sql_query(sql_query)
    logger.info("Executed SQL query", extra={
        "rows": 0 if result is None else getattr(result, "shape", [0])[0]
    })

    if result is None or result.empty:
        logger.info("Query returned no data")
        return {"error": "❌ Query returned no data"}

    # ✅ Clean result
    df_clean = result.replace([np.inf, -np.inf], np.nan).fillna("null")
    response = {
        "sql_query": sql_query,
        "top_rows": df_clean.head(10).to_dict(orient="records")
    }

    # ✅ Detect output format
    output_format = detect_output_format(question)
    logger.info("Detected output format", extra={"output_format": output_format})

    # ✅ Detect if visualization is requested
    visual_keywords = ["chart", "graph", "visual", "visualize"]
    include_charts = any(k in question.lower() for k in visual_keywords)

    output_path = None

    if output_format == "ppt":
        output_path = generate_ppt(question, df_clean, include_charts=include_charts)
    elif output_format == "excel":
        output_path = generate_excel(df_clean, question, include_charts=include_charts)
    elif output_format == "word":
        output_path = generate_word(df_clean, question, include_charts=include_charts)
    
    # ✅ Upload PPT/Excel/Word to external API
    if output_path:
        # ✅ Upload file
        response[f"{output_format}_path"] = output_path
        
        api_root = API_ROOT 
        
        # 1) POST metadata as JSON to PostSavePPTDetailsV2
        try:
            save_url = f"{api_root}PostSavePPTDetailsV2"
            metadata = {
                "FileName": encrypted_filename,
                "CreatedBy": created_by,
                "Date": datetime.now().strftime('%Y-%m-%d'),
            }
            # FIX 1: Pass FileName and CreatedBy as URL parameters (safer for APIs)
            save_params = {"FileName": encrypted_filename, "CreatedBy": created_by}
            
            logger.info("handle_agent_request: posting metadata", extra={"url": save_url, "params": save_params})
            
            # Using 'params' to ensure query string includes required fields
            save_resp = requests.post(save_url, json=metadata, params=save_params, timeout=REQUEST_TIMEOUT)
            
            response["metadata_status"] = save_resp.status_code
            response["metadata_response"] = save_resp.text[:1000]
            if save_resp.status_code != 200:
                response.setdefault("upload_warnings", []).append(f"Metadata save failed: {save_resp.status_code}")
                logger.warning("handle_agent_request: metadata save non-200", extra={"status": save_resp.status_code, "text": save_resp.text[:300]})
        except Exception as e:
            logger.exception("handle_agent_request: metadata POST failed")
            response["metadata_error"] = str(e)
            
        # 2) Upload file via multipart/form-data to UpdatePptFileV2
        try:
            if not os.path.exists(output_path):
                response["upload_status"] = f"Upload failed: generated file not found at {output_path}"
                logger.error("handle_agent_request: generated file not found", extra={"path": output_path})
            else:
                filtered_obj = {"slide": 1, "title": "Auto-generated Slide", "data": question}
                file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(output_format, "dat")
                filename_with_ext = f"{encrypted_filename}.{file_ext}"
                
                mime_map = {
                    "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                }
                mimetype = mime_map.get(output_format, "application/octet-stream")

                with open(output_path, "rb") as f:
                    files = {"file": (filename_with_ext, f, mimetype)}
                    
                    data_fields = {
                        "content": json.dumps({"content": [filtered_obj]}),
                    }

                    upload_url = f"{api_root}UpdatePptFileV2"
                    # FIX 2: Explicitly pass FileName and CreatedBy as URL parameters using 'params'
                    upload_params = {"FileName": encrypted_filename, "CreatedBy": created_by}
                    
                    logger.info("handle_agent_request: uploading file", extra={"url": upload_url, "params": upload_params, "filename": filename_with_ext})
                    
                    # Using 'params' argument for URL query string
                    upload_response = requests.post(
                        upload_url, 
                        data=data_fields, 
                        files=files, 
                        params=upload_params, # <-- FIX applied here
                        timeout=60
                    ) 

                    response["upload_code"] = upload_response.status_code
                    response["upload_response"] = upload_response.text[:2000]
                    if upload_response.status_code == 200:
                        response["upload_status"] = f"✅ {output_format.upper()} uploaded successfully"
                    else:
                        response["upload_status"] = f"❌ Upload failed: {upload_response.status_code}"
                        logger.warning("handle_agent_request: file upload failed", extra={"status": upload_response.status_code, "text": upload_response.text[:500]})
        except Exception as e:
            logger.exception("handle_agent_request: Exception during file upload or processing")
            response["upload_status"] = f"❌ Upload error: {str(e)}"
    
    # --- FIX: Removed redundant/misplaced empty except block ---

    logger.info("handle_agent_request: complete", extra={"has_output": bool(output_path), "output_format": output_format})
    return serialize(response)
  


# ✅ Test Agent
async def test_agent_response(agent_config: AgentConfig, structured_schema, sample_data, question
):
    #Convert dict to Pydantic model
    agent_config = AgentConfig(**agent_config)

    agent_name = agent_config.name
    question = question or (agent_config.sample_prompts[0] if agent_config.sample_prompts else "Give a summary of the data")

    sql_query = generate_sql_query(question, structured_schema)
    if not is_sql_read_only(sql_query):
        logger.warning("test_agent_response: non read-only SQL generated; rejecting")
        return {"error": "❌ Generated SQL is not read-only and was blocked by guardrails"}
    sql_query = enforce_sql_row_limit(sql_query)
    logger.info("test_agent_response: executing SQL", extra={"agent_name": agent_name})
    df = execute_sql_query(sql_query)

    if df is None or df.empty:
        logger.info("test_agent_response: no data returned")
        return {"error": "❌ No data returned"}

    df_clean = df.replace([np.inf, -np.inf], np.nan).fillna("null")

    # ✅ Generate a single, comprehensive response
    agent_response_content = generate_direct_response(question, df_clean)

    tone_prefix = f"Hello! I'm {agent_name}, your {agent_config.role}.\nUsing a {agent_config.tone} tone:"
    final_response = f"{tone_prefix}\n\n{agent_response_content}"
    insights, recs = generate_insights(df_clean)

    # --- FIX: Removed redundant/misplaced line: tone_prefix = f"Hello! I'm {agent_name}, ..." ---

    logger.info("test_agent_response: success", extra={"rows": df_clean.shape[0]})
    return {
       # "sql_query": sql_query,
        "top_rows": df_clean.head(10).to_dict(orient="records"),
        "insights": insights,
        "recommendations": recs,
        "agent_response": final_response
    }


def publish_agent (agent_name):
    try:
        if not agent_name:
            return {"error": "Missing 'agent_name'"}

        # 1. Get all agents
        logger.info("publish_agent: fetching all agents")
        get_response = requests.get(GET_ALL_AGENTS_URL, timeout=REQUEST_TIMEOUT)
        if get_response.status_code != 200:
            return {"error": f"Failed to fetch agents. Status code: {get_response.status_code}"}

        agents = get_response.json().get("Table", [])
        normalized_agents = [{k.lower(): v for k, v in agent.items()} for agent in agents]

        # 2. Find agent by name
        matching_index = next(
            (i for i, agent in enumerate(normalized_agents)
             if agent.get("name", "").lower() == agent_name.lower()),
            None
        )
        if matching_index is None:
            return {"error": f"Agent '{agent_name}' not found."}

        # 3. Get original agent details
        original_agent = agents[matching_index]

        # 4. Build payload — Published = True
        payload = {
            "ExistingAgentName": agent_name,
            "NewAgentName": original_agent.get("Name", ""),
            "ExistingRole": original_agent.get("Role", ""),
            "NewRole": original_agent.get("Role", ""),
            "ExistingPurpose": original_agent.get("Purpose", ""),
            "NewPurpose": original_agent.get("Purpose", ""),
            "ExistingInstruction": original_agent.get("Instructions", ""),
            "Instruction": original_agent.get("Instructions", ""),
            "Existingcapabilities": original_agent.get("Capabilities", ""),
            "Capabilities": original_agent.get("Capabilities", ""),
            "Published": "True"  # ✅ Set Published to True
        }

        # ✅ Log payload
        logger.info("publish_agent: payload prepared", extra={"payload_keys": list(payload.keys())})

        # 5. Send POST request
        post_response = requests.post(PUBLISH_AGENT_URL, json=payload, timeout=REQUEST_TIMEOUT)

        logger.info("publish_agent: response", extra={"status": post_response.status_code})

        # 6. Handle response
        if post_response.status_code == 200 and post_response.text.strip().lower() not in ["internal server error", ""]:
            try:
                response_json = post_response.json()
                return {
                    "message": f"✅ Agent '{agent_name}' published successfully",
                    "updated_config": response_json
                }
            except Exception as e:
                return {
                    "message": f"✅ Agent '{agent_name}' published successfully (non-JSON response)",
                    "updated_config": {
                        "raw_response": post_response.text,
                        "parse_error": str(e)
                    }
                }
        else:
            return {
                "error": f"❌ Failed to publish agent. Status code: {post_response.status_code}",
                "details": post_response.text
            }

    except Exception as e:
        logger.exception("publish_agent: exception")
        return {"error": f"❌ Exception occurred: {str(e)}"}
    
def schedule_agent(data):
    agent_name = data.get("name")
    path = f"{AGENT_DIR}/{agent_name}.json"
    if not os.path.exists(path):
        return {"error": "❌ Agent config not found"}
    with open(path, "r+") as f:
        config = json.load(f)
        config["schedule_enabled"] = True
        config["frequency"] = data.get("frequency")
        config["time"] = data.get("time")
        config["output_method"] = data.get("output_method")
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()
    return {"message": "✅ Agent scheduled"}


def load_agent_config(name: str) -> AgentConfig:
    """Load agent configuration from database with enhanced field handling"""
    try:
        logger.info("load_agent_config: fetching agent", extra={"agent_name": name})
        resp = requests.get(API_URL, params={"AgentName": name}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("Table", [])
    except requests.exceptions.RequestException as e:
        logger.exception("Error fetching agent details")
        return None

    if not data:
        logger.warning("No agent data returned", extra={"agent_name": name})
        return None

    # Group by name to handle multiple versions
    grouped = defaultdict(list)
    for record in data:
        agent_name = record.get("Name")
        if agent_name:
            grouped[agent_name].append(record)

    entries = grouped.get(name)
    if not entries:
        return None

    # Find the most recent entry
    for rec in entries:
        try:
            rec["_parsed_time"] = datetime.fromisoformat(rec.get("Time"))
        except (ValueError, TypeError):
            # Fallback to current time if parsing fails
            rec["_parsed_time"] = datetime.now()

    latest = sorted(entries, key=lambda x: x["_parsed_time"], reverse=True)[0]
    latest.pop("_parsed_time", None)

    # Normalize published to boolean
    published_raw = latest.get("Published", False)
    published = str(published_raw).lower() == "true"

    # Enhanced field normalization with better error handling
    transformed = {
        "name": latest.get("Name", ""),
        "role": latest.get("Role", ""),
        "purpose": latest.get("Purpose", ""),
        "instructions": _ensure_list(latest.get("Instructions")),
        "capabilities": _ensure_list(latest.get("Capabilities")),
        "welcome_message": latest.get("WelcomeMessage") or "",
        "knowledge_base": _ensure_list(latest.get("KnowledgeBase")),
        "sample_prompts": _ensure_list(latest.get("SamplePrompts")),
        "tone": latest.get("Tone", "neutral"),
        "published": published
    }

    # Validate required fields
    if not transformed["name"] or not transformed["purpose"]:
        logger.warning("Missing required fields in agent config", extra={"agent_name": name})
        return None

    logger.info("load_agent_config: success", extra={
        "agent_name": transformed.get("name"),
        "published": transformed.get("published"),
        "capabilities_count": len(transformed.get("capabilities", []))
    })
    
    return AgentConfig(**transformed)


# Ensure we correctly convert list to comma-separated string
def list_to_str(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value or ""

def edit_agent_config(existing_name, new_data):
    try:
        new_name = new_data.get("name")
        new_role = new_data.get("role")
        new_purpose = new_data.get("purpose")
        new_instruction = new_data.get("instruction")
        new_capabilities = new_data.get("capabilities")

        if not existing_name:
            return {"error": "Missing 'ExistingAgentName'"}

        # Step 1: Fetch agents
        get_response = requests.get(GET_ALL_AGENTS_URL, timeout=REQUEST_TIMEOUT)
        if get_response.status_code != 200:
            return {"error": f"Failed to fetch agents. Status code: {get_response.status_code}"}

        agents = get_response.json().get("Table", [])
        normalized_agents = [{k.lower(): v for k, v in agent.items()} for agent in agents]

        # Step 2: Find the existing agent
        matching_index = next(
            (i for i, agent in enumerate(normalized_agents)
             if agent.get("name", "").lower() == existing_name.lower()),
            None
        )
        if matching_index is None:
            return {"error": f"Agent '{existing_name}' not found."}

        original_agent = agents[matching_index]

        # Step 3: Build Payload
        # Use new data if provided, otherwise fallback to existing data
        payload = {
            "ExistingAgentName": existing_name,
            "NewAgentName": new_name or original_agent.get("Name", ""),
            "ExistingRole": original_agent.get("Role", ""),
            "NewRole": new_role or original_agent.get("Role", ""),
            "ExistingPurpose": original_agent.get("Purpose", ""),
            "NewPurpose": new_purpose or original_agent.get("Purpose", ""),
            "Published": original_agent.get("Published", "False"),
            # Ensure Instruction and Capabilities are correctly formatted as strings for the API
            "ExistingInstruction": list_to_str(new_data.get("ExistingInstruction") or original_agent.get("Instructions")),
            "Instruction": list_to_str(new_data.get("Instruction") or original_agent.get("Instructions")),
            "Existingcapabilities": list_to_str(new_data.get("Existingcapabilities") or original_agent.get("Capabilities")),
            "Capabilities": list_to_str(new_data.get("Capabilities") or original_agent.get("Capabilities"))
        }

        logger.info("edit_agent_config: payload prepared", extra={"existing_name": existing_name})
        
        post_response = requests.post(EDIT_AGENT_URL, json=payload, timeout=REQUEST_TIMEOUT)

        logger.info("edit_agent_config: response", extra={"status": post_response.status_code})

        if post_response.status_code == 200 and post_response.text.strip().lower() not in ["internal server error", ""]:
            try:
                return {
                    "message": "✅ Agent updated successfully",
                    "updated_config": post_response.json()
                }
            except Exception:
                 return {
                    "message": "✅ Agent updated successfully (non-JSON response)",
                    "updated_config": {"raw_response": post_response.text}
                }
        else:
            return {
                "error": f"❌ Failed to update agent. Status code: {post_response.status_code}",
                "details": post_response.text
            }

    except Exception as e:
        logger.exception("edit_agent_config: exception")
        return {"error": f"❌ Exception occurred: {str(e)}"}