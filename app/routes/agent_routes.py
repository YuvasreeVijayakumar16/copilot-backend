from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.utils.schema_reader import get_schema_and_sample_data, get_db_schema
from app.utils.llm_validator import validate_purpose_and_instructions
from app.utils.agent_builder import VALID_ROLES, generate_sample_prompts
from app.services.agent_servies import save_agent_config, ALLOWED_CAPABILITIES
from app.services.agent_servies import test_agent_response
from app.models.agent import AgentConfig
from app.utils.gpt_utils import serialize
from fastapi.encoders import jsonable_encoder
import traceback
from uuid import uuid4
import requests
from app.services.agent_servies import (
    edit_agent_config,
    publish_agent,
    test_agent_response,
    load_agent_config
)
from app.services.agent_servies import handle_agent_request
import logging
import math
import numpy as np
import ast
import json

logger = logging.getLogger("app.routes.agent_routes")
from app.agents.autogen_orchestrator import run_autogen_orchestration
from app.agents.autogen_manager import AgentManager
router = APIRouter()


def sanitize_for_json(obj):
    """Recursively sanitize a Python object for JSON encoding.

    Replaces non-finite floats (inf, -inf, nan) with None and converts
    numpy scalars/arrays to native Python types.
    """
    # primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj

    # numpy scalar
    if isinstance(obj, (np.generic,)):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            return None

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # dict
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    # pandas objects or objects exposing tolist
    try:
        if hasattr(obj, "tolist"):
            return sanitize_for_json(obj.tolist())
    except Exception:
        pass

    # fallback to string representation
    try:
        return str(obj)
    except Exception:
        return None



# In-memory session tracking (temporary store for demo purposes)
user_threads = {}
user_collected_fields = {}

@router.post("/agent-message")
async def agent_message(request: Request):
    try:
        data = await request.json()
        logger.info("agent_message: received", extra={
            "has_user_id": bool(data.get("user_id")),
            "message_len": len((data.get("message") or ""))
        })
        # AutoGen short-circuit
        if str(data.get("engine", "")).lower() == "autogen":
            question = data.get("message", "")
            output_format = data.get("output_format")
            logger.info("agent_message: using AutoGen orchestration", extra={"output_format": output_format})
            result = run_autogen_orchestration(question, output_format=output_format)
            return JSONResponse(sanitize_for_json(result))
        user_id = data.get("user_id")
        message = data.get("message", "")

        if not user_id or not message:
            logger.warning("agent_message: missing user_id or message")
            return JSONResponse({"error": "Missing user_id or message"}, status_code=400)

        # Init user thread
        if user_id not in user_threads:
            user_threads[user_id] = []
            user_collected_fields[user_id] = {
                "name": None,
                "role": None,
                "purpose": None,
                "instructions": None,
                "capabilities": None,
                "welcome_message": None,
            }

        thread = user_threads[user_id]
        collected = user_collected_fields[user_id]
        thread.append({"user": message})

        # Collect agent data step-by-step
        #if not collected["name"]:
         #   collected["name"] = message
          #  return JSONResponse({"message": "What is the agent's role?"})
        if not collected["name"]:
            collected["name"] = message
            logger.info("agent_message: asking for role options")
            return JSONResponse({
             "message": "Choose role",
            "Roles": VALID_ROLES
         })
        # Check if the role is valid
        # If not, return an error message with valid options
        elif not collected["role"]:
            if message not in VALID_ROLES:
                logger.warning("agent_message: invalid role", extra={"role": message})
                return JSONResponse({
                    "error": f"Invalid role: '{message}'. Please provide a valid role from the following options: {', '.join(VALID_ROLES)}",
                    "allowed_roles": VALID_ROLES
                }, status_code=400)
            
            collected["role"] = message
            logger.info("agent_message: role accepted; asking purpose")
            return JSONResponse({"message": "What is the purpose of this agent?"})

        # === PURPOSE VALIDATION MOVED HERE ===
        elif not collected["purpose"]:
            # Perform validation immediately after getting the purpose
            # Note: The validation function often requires instructions as well.
            # Here, we'll assume a simplified validation that only checks the purpose field.
            # You will need to adjust your `validate_purpose_and_instructions` function.
            
            # Placeholder for purpose-only validation
            structured_schema, _, sample_data = get_schema_and_sample_data()
            validation = validate_purpose_and_instructions(
                message, "", structured_schema, sample_data
            )
            
            if not validation.get("purpose_valid", False):
                logger.info("agent_message: invalid purpose provided")
                # Only reset the purpose field, so the user can re-enter it.
                collected["purpose"] = None
                return JSONResponse({
                    "error": "Invalid purpose. Must relate to querying, analyzing, summarizing, or reporting on data. Please provide a new purpose.",
                }, status_code=400)

            # If validation passes, save the purpose and ask for the next field.
            collected["purpose"] = message
            logger.info("agent_message: purpose accepted; asking instructions")
            return JSONResponse({"message": "What are the detailed instructions for the agent?"})

        # ✅ 4. Get Agent Instructions
        elif not collected["instructions"]:
            collected["instructions"] = message
            logger.info("agent_message: asking capabilities")
            return JSONResponse({
        "message": "List the agent's capabilities (comma-separated).",
        "allowed_capabilities": ALLOWED_CAPABILITIES
    })
            
        #elif not collected.get("capabilities"):
            #collected["capabilities"] = [cap.strip() for cap in message.split(",")]
            #return JSONResponse({"message": "What welcome message should the agent greet users with?"})

        # ✅ 5. Get Agent Capabilities
   # ✅ 5. Get Agent Capabilities
        elif not collected.get("capabilities"):
    
    # Check if the message is a string
    # 👇 NEW/CHANGED LINE
            if isinstance(message, str):
        # Process the string as a comma-separated list
        # 👇 NEW/CHANGED LINE
                user_capabilities = [cap.strip() for cap in message.split(",")]
    
    # Check if the message is already a list (from Postman)
    # 👇 NEW/CHANGED LINE
            elif isinstance(message, list):
        # Use the list directly and strip any whitespace from each item
        # 👇 NEW/CHANGED LINE
                    user_capabilities = [cap.strip() for cap in message]
    
    # Handle other unexpected data types
    # 👇 NEW/CHANGED LINE
            else:
                logger.warning("agent_message: invalid capabilities type", extra={"type": type(message).__name__})
                return JSONResponse({"error": "Invalid data type for capabilities."}, status_code=400)
    
    # Check if every user-provided capability is in the allowed list
            for capability in user_capabilities:
                    if capability not in ALLOWED_CAPABILITIES:
                        logger.warning("agent_message: invalid capability", extra={"capability": capability})
                        return JSONResponse({
                "error": f"Invalid capability: '{capability}'. Please provide a valid capability from the following options: {', '.join(ALLOWED_CAPABILITIES)}",
                "allowed_capabilities": ALLOWED_CAPABILITIES
            }, status_code=400)
    
    # If all capabilities are valid, assign them and proceed
            collected["capabilities"] = user_capabilities
            logger.info("agent_message: capabilities accepted; asking welcome message")
            return JSONResponse({"message": "What welcome message should the agent greet users with?"})

# ✅ Welcome Message Validation
        elif not collected.get("welcome_message"):
             collected["welcome_message"] = message

          
        structured_schema, _, sample_data = get_schema_and_sample_data()
        validation = validate_purpose_and_instructions(
        collected["purpose"], collected["instructions"], structured_schema, sample_data
            )

        if validation.get("invalid_instructions"):
                logger.info("agent_message: invalid instructions")
                # Clear invalid instructions and dependents.
                collected["instructions"] = None
                collected["capabilities"] = None
                collected["welcome_message"] = None
                return JSONResponse({
                    "error": "Invalid instructions. Please provide new instructions.",
                    "invalid_instructions": validation["invalid_instructions"]
                }, status_code=400)

            
            

            # ✅ Final config
        agent_config = {
                "name": collected["name"],
                "role": collected["role"],
                "purpose": collected["purpose"],
                "instructions": [collected["instructions"]],
                "sample_prompts": generate_sample_prompts(collected["purpose"], collected["role"]),
                "tone": "friendly",
                "knowledge_base": [],
                "welcome_message": collected["welcome_message"],
                "capabilities": collected["capabilities"],
                "system_prompt": "You are a SQL assistant. Return only the query without explanation.",
                "frequency": "once",
                "time": "09:00",
                "schedule_enabled": True,
                "output_method": "chat",
                "published": False
            }

        agent_model = AgentConfig(**agent_config)
        save_agent_config(agent_model)

            # ✅ Sync to external API
        try:
                api_url = "https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/SaveAgentDetails"
                payload = {
                    "Name": agent_model.name,
                    "Role": agent_model.role,
                    "Purpose": agent_model.purpose,
                    "Instructions": agent_model.instructions,
                    "Capabilities": agent_model.capabilities,
                    "WelcomeMessage": agent_model.welcome_message,
                    "Tone": agent_model.tone,
                    "KnowledgeBase": agent_model.knowledge_base,
                    "SamplePrompts": agent_model.sample_prompts,
                    "ScheduleEnabled": agent_model.schedule_enabled,
                    "Frequency": agent_model.frequency,
                    "Time": agent_model.time,
                    "OutputMethod": agent_model.output_method,
                    "Published": agent_model.published
                }

                logger.info("agent_message: syncing agent to external API")
                api_response = requests.post(api_url, json=payload)

                try:
                    api_body = api_response.json()
                except ValueError:
                    api_body = api_response.text or "No response body"

                db_status = {
                    "code": api_response.status_code,
                    "body": api_body
                }

        except Exception as e:
                db_status = {
                    "code": "error",
                    "body": f"❌ API call failed: {str(e)}"
                }

            # ✅ Optionally test agent behavior
            #result = await test_agent_response(agent_config, structured_schema, sample_data)

            # ✅ Reset after success
        user_threads[user_id] = []
        user_collected_fields[user_id] = {}
        logger.info("agent_message: agent created", extra={"status": api_response.status_code})

        return JSONResponse(content=jsonable_encoder({
                "message": "Agent created and validated successfully!",
                "agent_config": agent_config,
                #"test_result": result,
                "sync_status": db_status
            }))

    except Exception as e:
        logger.exception("agent_message: exception")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/agent/edit")
async def edit_agent(request: Request):
    try:
        data = await request.json()
        logger.info("edit_agent: received", extra={"keys": list(data.keys())})

        # Extract existing_name and new_data from incoming request
        existing_name = data.get("ExistingAgentName")
        new_data = {
            "name": data.get("NewAgentName"),
            "role": data.get("NewRole"),
            "purpose": data.get("NewPurpose"),
            "instruction" : data.get("Instruction"),
            "capabilities" : data.get("Capabilities")
        }

        if not existing_name or not all(new_data.values()):
            return JSONResponse(
                content={"error": "Missing required fields in the request"},
                status_code=400
            )

        # Now call your edit_agent_config with 2 arguments
        result = edit_agent_config(existing_name, new_data)

        if not isinstance(result, dict): result = {"message": str(result)}

        return JSONResponse(content={
            "message": "✅ Agent updated successfully via API",
            "updated_config": result
        }, status_code=200)

    except Exception as e:
        logger.exception("edit_agent: exception")
        return JSONResponse(content={
            "message": "❌ Exception occurred while editing agent.",
            "details": str(e)
        }, status_code=500)


@router.post("/agent/publish")
async def publish_existing_agent(request: Request):
    try:
        data = await request.json()
        agent_name = data.get("name")  # Expecting {"name": "Loki"}

        if not agent_name:
            return JSONResponse({"error": "Missing 'name' in request body"}, status_code=400)

        logger.info("publish_existing_agent: publishing", extra={"agent_name": agent_name})
        result = publish_agent(agent_name)  # ✅ Pass only the name
        return JSONResponse(result)

    except Exception as e:
        logger.exception("publish_existing_agent: exception")
        return JSONResponse(
            {"error": f"❌ Exception occurred while publishing agent: {str(e)}"},
            status_code=500
        )

@router.post("/agent/test")
async def test_existing_agent(request: Request):
    data = await request.json()
    name = data.get("name")
    question = data.get("question")
    created_by = data.get("created_by")
    encrypted_filename = data.get("encrypted_filename")
    # Auto-select agent if not provided
    if not name:
        mgr = AgentManager()
        agents = mgr.discover_all_agents()
        chosen = mgr.route(question or "", agents)
        name = chosen.get("name") if isinstance(chosen, dict) else None
    logger.info("test_existing_agent: using AutoGen orchestration", extra={"agent_name": name})
    result = run_autogen_orchestration(
        question,
        agent_name=name,
        created_by=created_by,
        encrypted_filename=encrypted_filename,
    )
    return JSONResponse(sanitize_for_json(result))




user_threads = {}  # Make sure this is defined globally somewhere

@router.post("/agent/play/{name}")
async def play_agent(name: str, request: Request):
    try:
        data = await request.json()
    except Exception:
        logger.warning("play_agent: invalid JSON body")
        return JSONResponse({"error": "Invalid or missing JSON body"}, status_code=400)

    user_id = data.get("user_id")
    message = data.get("message", "").strip()

    if not user_id or not message:
        return JSONResponse({"error": "Missing user_id or message"}, status_code=400)

    agent_config = load_agent_config(name)
    if not agent_config:
        return JSONResponse({"error": "Agent not found"}, status_code=404)

    if not agent_config.published:
        return JSONResponse({"error": "Agent not published"}, status_code=400)

    # Initialize thread for user if not exists
    if user_id not in user_threads:
        user_threads[user_id] = []

    # Append user message
    user_threads[user_id].append({"user": message})

    # Use official AutoGen orchestration for chat
    logger.info("play_agent: invoking AutoGen orchestration", extra={"agent_name": name, "user_id": user_id})
    created_by = data.get("created_by")
    encrypted_filename = data.get("encrypted_filename")
    result = run_autogen_orchestration(message, agent_name=name, created_by=created_by, encrypted_filename=encrypted_filename)

    agent_reply = result.get("answer") or "Sorry, no response generated."
    user_threads[user_id].append({"agent": agent_reply})

    logger.info("play_agent: response ready", extra={"has_agent_response": bool(result.get("answer"))})
    return JSONResponse(sanitize_for_json(result))


@router.post("/reset-conversation")
async def reset_conversation(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")

        if not user_id:
            return JSONResponse({"error": "Missing user_id"}, status_code=400)

        # Reset the user's conversation state
        if user_id in user_threads:
            del user_threads[user_id]
        if user_id in user_collected_fields:
            del user_collected_fields[user_id]
            
        logger.info("reset_conversation: reset", extra={"user_id": user_id})
        return JSONResponse({"message": "Conversation has been reset."}, status_code=200)

    except Exception as e:
        logger.exception("reset_conversation: exception")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/agent/autogen")
async def autogen_orchestrate(request: Request):
    data = await request.json()
    question = data.get("question", "")
    logger.info("autogen_orchestrate: received", extra={"qlen": len(question)})
    result = run_autogen_orchestration(question)
    return JSONResponse(sanitize_for_json(result))


@router.post("/agent/manager/plan_run")
async def plan_and_run_manager(request: Request):
    data = await request.json()
    task = data.get("task", "").strip()
    agents = data.get("agents")  # optional now
    created_by = data.get("created_by")
    encrypted_filename = data.get("encrypted_filename")
    if not task:
        return JSONResponse({"error": "Missing 'task'"}, status_code=400)
    logger.info("plan_and_run_manager: received", extra={"task_len": len(task), "agents": (len(agents) if isinstance(agents, list) else 0)})
    mgr = AgentManager()
    # Execute and, if a single agent is chosen in steps, pass upload metadata through
    result = mgr.plan_and_run(task, agents)
    # Best-effort propagate upload after the fact is complex; rely on orchestration handling per step.
    return JSONResponse(sanitize_for_json(result))

# agent_routes.py (New endpoints)
@router.post("/agent/orchestrate")
async def orchestrate_agents(request: Request):
    """Execute multi-agent workflow for a task"""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        body = await request.body()
        data = ast.literal_eval(body.decode())
    task = data.get("task")
    session_id = data.get("session_id", str(uuid4()))

    if not task:
        return JSONResponse({"error": "Task is required"}, status_code=400)

    # Optional metadata for file upload
    created_by = data.get("created_by")
    encrypted_filename = data.get("encrypted_filename")

    # Optional metadata
    output_format = data.get("output_format")

    # Initialize agent manager
    manager = AgentManager()

    # Build initial context forwarded into agents/orchestrator
    ctx = {}
    if output_format:
        ctx["output_format"] = output_format
    if created_by:
        ctx["created_by"] = created_by
    if encrypted_filename:
        ctx["encrypted_filename"] = encrypted_filename

    # Plan and execute workflow, passing initial context
    result = manager.plan_and_run(task, ctx)
    safe_result = sanitize_for_json(result)

    # Extract agents used from the steps
    agents_used = []
    steps = safe_result.get("steps", [])
    for step in steps:
        agent_name = step.get("agent")
        if agent_name and agent_name not in agents_used:
            agents_used.append(agent_name)

    # Extract table_data and file_upload_status from the last relevant step
    table_data = None
    file_upload_status = None
    for step in reversed(steps):
        res = step.get("result", {})
        if res.get("preview_rows") and table_data is None:
            table_data = res["preview_rows"]
        if res.get("upload_status") and file_upload_status is None:
            file_upload_status = res["upload_status"]

    return JSONResponse({
        "session_id": session_id,
        "task": task,
        "agents_used": agents_used,
        "table_data": table_data,
        "file_upload_status": file_upload_status,
        "result": safe_result
    })

@router.post("/agent/chat")
async def multi_agent_chat(request: Request):
    """Continue conversation with multi-agent context"""
    data = await request.json()
    message = data.get("message")
    session_id = data.get("session_id")
    
    if not message or not session_id:
        return JSONResponse({"error": "Message and session_id are required"}, status_code=400)
    
    # Initialize agent manager
    manager = AgentManager()
    
    # Get previous context
    context = manager.context_history.get(session_id, {})
    
    # Route message to appropriate agent
    agents = manager.discover_agents()
    agent = manager.route_task(message, agents)
    
    # Execute with context
    result = run_autogen_orchestration(
        question=message,
        agent_name=agent.get("name"),
        context=context
    )
    
    # Update context
    if result.get("answer"):
        context["last_response"] = result["answer"]
        manager.context_history[session_id] = context
    
    return JSONResponse({
        "session_id": session_id,
        "agent": agent.get("name"),
        "role": agent.get("role"),
        "response": sanitize_for_json(result)
    })

@router.get("/agent/context/{session_id}")
async def get_session_context(session_id: str):
    """Get conversation context for a session"""
    manager = AgentManager()
    context = manager.context_history.get(session_id, {})
    return JSONResponse({
        "session_id": session_id,
        "context": context
    })