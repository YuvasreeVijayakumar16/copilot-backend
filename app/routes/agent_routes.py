from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.utils.schema_reader import get_schema_and_sample_data, get_db_schema
from app.utils.llm_validator import validate_purpose_and_instructions
from app.utils.agent_builder import VALID_ROLES,generate_sample_prompts
from app.services.agent_servies import save_agent_config, ALLOWED_CAPABILITIES
from app.models.agent import AgentConfig
from fastapi.encoders import jsonable_encoder
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from app.db.sql_connection import get_connection_string
from pydantic import BaseModel
from app.agents.autogen_manager import AgentManager
from app.agents.autogen_orchestrator import run_autogen_orchestration
from app.utils.gpt_utils import serialize
from uuid import uuid4
from app.services.agent_servies import (
    edit_agent_config,
    publish_agent,
    test_agent_response,
    load_agent_config
)
from app.services.agent_servies import handle_agent_request
import requests
import math
import numpy as np
import ast
import json
logger = logging.getLogger("app.routes.agent_routes")
router = APIRouter()


# ===============================
# 🔐 GLOBAL PROMPT SECURITY LAYER
# ===============================

BLOCKED_KEYWORDS = [
    "conversation",
    "other users",
    "previous user",
    "system prompt",
    "openai",
    "model details",
    "tools",
    "plugins",
    "api key",
    "internal configuration",
    "how to call",
    "reveal prompt",
    "training data",
    "trained on",
    "knowledge base",
    "what were you trained",
    "list all documents",
    "file paths",
    "connected with",
    "internal files",
    "source data"
   
]

ALLOWED_TOPICS = [
    "inventory",
    "material",
    "supply",
    "lead time",
    "purchase",
    "manufacturing",
    "demand",
    "forecast",
    "capacity",
    "capex",
    "PO",
    "order",
    "reorder"
    "stockout",
    "overstock",
    "understock",
    "fillrate",
    "SLA met",
    "delivery",
    "turnover",
    "backorder",
    "order history",
    "demand history",
    "forecast accuracy",
    "outstanding order"

]


import re

SENSITIVE_PATTERNS = [
    r"(?i)list.*(document|file|path|data)",
    r"(?i)what.*trained",
    r"(?i)search.*knowledge",
    r"(?i)show.*internal",
    r"(?i)reveal.*data",
]

def contains_sensitive_probe(text: str) -> bool:
    text = text or ""
    lowered = text.lower()

    if any(keyword in lowered for keyword in BLOCKED_KEYWORDS):
        return True

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def is_domain_query(text: str) -> bool:
    text = (text or "").lower()
    return any(topic in text for topic in ALLOWED_TOPICS)




def validate_prompt_security(text: str):
    if not text:
        return JSONResponse({"error": "Invalid request."}, status_code=403)

    lowered = text.lower()

    # Explicit block (case-insensitive via lowercase)
    for keyword in BLOCKED_KEYWORDS:
        if keyword.lower() in lowered:
            return JSONResponse({"error": "Invalid request."}, status_code=403)

    # Block sensitive regex patterns
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return JSONResponse({"error": "Invalid request."}, status_code=403)

    # Allow only domain-based queries
    if not any(topic.lower() in lowered for topic in ALLOWED_TOPICS):
        return JSONResponse(
            {"error": "Only supply-chain analytics queries are allowed."},
            status_code=403
        )

    return None


 
# Session tracking per user
user_threads = {}
user_collected_fields = {}
 
 
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
 
# -------------------------------------------
#  ROLE LOOKUP FUNCTION (Dynamic per CUID)
# -------------------------------------------
def get_roles_for_cuid(cuid: str):
    try:
        engine = create_engine(get_connection_string())
 
        admin_query = text("""
            SELECT TOP 1 ur.User_Role
            FROM UserSettings us
            LEFT JOIN user_role ur ON us.UserNT = ur.UserNT
            WHERE us.UserNT = :cuid
            AND ur.User_Role = 'Supply Chain Transformation Manager (admin)'
        """)
 
        all_roles_query = text("SELECT DISTINCT Role FROM Agent_details WHERE Role IS NOT NULL")
 
        user_roles_query = text("""
            SELECT DISTINCT ur.User_Role
            FROM UserSettings us
            LEFT JOIN user_role ur ON us.UserNT = ur.UserNT
            WHERE us.UserNT = :cuid AND ur.User_Role IS NOT NULL
        """)
 
        with engine.connect() as conn:
            is_admin = conn.execute(admin_query, {"cuid": cuid}).fetchone()
 
            if is_admin:
                logger.info(f"{cuid} is Admin → All roles returned")
                result = conn.execute(all_roles_query).fetchall()
            else:
                logger.info(f"{cuid} is Normal User → Assigned roles returned")
                result = conn.execute(user_roles_query, {"cuid": cuid}).fetchall()
 
        roles = [row[0] for row in result] if result else ["Default Role"]
        logger.info(f"Roles fetched for {cuid}: {roles}")
        return roles
 
    except Exception as e:
        logger.error(f"Role fetch failed for CUID={cuid}: {e}")
        return ["Default Role"]
 
# -------------------------------------------
# POST: /agent-message
# -------------------------------------------
@router.post("/agent-message")
async def agent_message(request: Request):
    try:
        data = await request.json()
        logged_cuid = data.get("logged_cuid")
        user_id = data.get("user_id")
        message = data.get("message", "")
 
        if not logged_cuid or not user_id or not message:
            return JSONResponse({"error": "logged_cuid, user_id, message required"}, status_code=400)
 
        dynamic_roles = get_roles_for_cuid(logged_cuid)
 
        # Initialize session
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
 
        # 1️⃣ Name
        if not collected["name"]:
            collected["name"] = message
            return JSONResponse({"message": "Role", "Roles": dynamic_roles})
 
        # 2️⃣ Role validation
        elif not collected["role"]:
            if message not in dynamic_roles:
                return JSONResponse({
                    "error": "Invalid role",
                    "allowed_roles": dynamic_roles
                }, status_code=400)
            collected["role"] = message
            return JSONResponse({"message": "What is the agent's purpose?"})
 
        # 3️⃣ Purpose validation
        elif not collected["purpose"]:
            schema, _, sample_data = get_schema_and_sample_data()
            validation = validate_purpose_and_instructions(message, "", schema, sample_data)
            if not validation.get("purpose_valid"):
                return JSONResponse({"error": "Invalid purpose"}, status_code=400)
            collected["purpose"] = message
            return JSONResponse({"message": "Enter detailed instructions"})
 
        # 4️⃣ Instructions
        elif not collected["instructions"]:
            collected["instructions"] = message
            return JSONResponse({
                "message": "Enter capabilities (comma-separated)",
                "allowed_capabilities": ALLOWED_CAPABILITIES
            })
 
        # 5️⃣ Capability validation
  
        elif not collected["capabilities"]:

            # If message is a single string → treat as comma separated list
            if isinstance(message, str):
                user_capabilities = [cap.strip() for cap in message.split(",")]

            # If message is already a list → treat directly
            elif isinstance(message, list):
                user_capabilities = [cap.strip() for cap in message]

            # Anything else → invalid
            else:
                return JSONResponse({"error": "Invalid data type for capabilities. Must be string or list."},
                                    status_code=400)

            # Validate against allowed capabilities
            for capability in user_capabilities:
                if capability not in ALLOWED_CAPABILITIES:
                    return JSONResponse({
                        "error": f"Invalid capability: '{capability}'.",
                        "allowed_capabilities": ALLOWED_CAPABILITIES
                    }, status_code=400)

            # Save & continue
            collected["capabilities"] = user_capabilities
            return JSONResponse({"message": "What welcome message should the agent greet users with?"})

                
        # 6️⃣ Final save
        elif not collected["welcome_message"]:
            collected["welcome_message"] = message
 
        # Validation done
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
            "schedule_enabled": True,
            "frequency": "once",
            "time": "09:00",
            "output_method": "chat",
            "published": False
        }
        agent_model = AgentConfig(**agent_config)
        save_agent_config(AgentConfig(**agent_config))
 
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
    
    security_error = validate_prompt_security(question)
    if security_error:
        return security_error



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
    security_error = validate_prompt_security(message)
    if security_error:
        return security_error
    result = run_autogen_orchestration(message, agent_name=name, created_by=created_by, encrypted_filename=encrypted_filename)
 
    #agent_reply = result.get("answer") or "Sorry, no response generated."
    if isinstance(result, dict) and result:
        # If result is a non-empty dictionary, safely extract the answer
        agent_reply = result.get("answer") or "Sorry, no response generated."
        has_answer = bool(result.get("answer"))
    else:
        # If result is None (due to unhandled exception) or not a dict
        agent_reply = "An internal error prevented the agent from responding."
        has_answer = False  # <-- MODIFIED: Create safe variable
        if result is None:
            result = {"error": agent_reply, "answer": agent_reply}
    user_threads[user_id].append({"agent": agent_reply})
 
    logger.info("play_agent: response ready", extra={"has_agent_response": has_answer})  # <-- MODIFIED: Use safe variable
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
    security_error = validate_prompt_security(question)
    if security_error:
        return security_error
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
    security_error = validate_prompt_security(task)
    if security_error:
        return security_error
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

    security_error = validate_prompt_security(task)
    if security_error:
        return security_error
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
        "combined_results": safe_result.get("combined_results"),
        "file_url": safe_result.get("file_url"),
        "steps": safe_result.get("steps"),
        "plan": safe_result.get("plan")
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
    security_error = validate_prompt_security(message)
    if security_error:
        return security_error

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
 
class UserRequest(BaseModel):
    UserNT: str
 
@router.post("/user-agents")
def get_user_agents(request: UserRequest):
 
    query = text("""
        SELECT
            us.User_Fullname,
            ur.User_Role,
            ad.Name AS Agent_Name,
            ad.Role AS Agent_Role
        FROM UserSettings us
        LEFT JOIN user_role ur
            ON us.UserNT = ur.UserNT
        LEFT JOIN Agent_details ad
            ON (
                ur.User_Role = 'Supply Chain Transformation Manager (admin)'
                OR ad.Role = ur.User_Role
            )
        WHERE us.UserNT = :UserNT;
    """)
 
    engine = create_engine(get_connection_string())
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"UserNT": request.UserNT})
 
    if df.empty:
        return {"error": "User not found"}
 
    user_fullname = df["User_Fullname"].iloc[0]
    user_role = df["User_Role"].iloc[0]
    agents = df["Agent_Name"].dropna().unique().tolist()
 
    return {
        "user_fullname": user_fullname,
        "user_role": user_role,
        "agents": agents
    }