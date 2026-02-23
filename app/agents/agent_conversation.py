from openai import OpenAI
from app.utils.llm_validator import validate_purpose_and_instructions
from app.utils.agent_builder import VALID_ROLES, generate_sample_prompts
from uuid import uuid4
from urllib.parse import urlparse # 🎯 NEW LINE

client = OpenAI()

# ==============================
# 🔹 AGENT NODE CLASS
# ==============================
class AgentNode:
    def __init__(self, name: str, purpose: str, config: dict):
        self.name = name
        self.purpose = purpose
        self.config = config

    def health_check(self) -> bool:
        """Simple placeholder for agent health check."""
        return True

    # --- ✅ FIX: prevent crash when emitting events ---
    def _emit_event(self, event_name: str, payload: dict, reward: float = 0.0):
        """Stub for lifecycle events; logs instead of emitting to message bus."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[Event:{event_name}] {self.name} -> {payload} | reward={reward}")

    # --- ✅ MAIN EXECUTION LOGIC ---
    def execute(self, task: str, context: dict = None) -> dict:
        from app.utils.ppt_generator import (
            generate_direct_response,
            generate_ppt_enhanced as generate_ppt,
            generate_excel,
            generate_word,
            generate_insights,
        )
        import pandas as pd
        import os, json, requests, logging
        from datetime import datetime
        from uuid import uuid4

        logger = logging.getLogger(__name__)
        

        def sanitize_output(text: str) -> str:
            if not isinstance(text, str):
                return text

            forbidden = [
                "system prompt",
                "openai",
                "model",
                "tools",
                "other users",
                "internal",
            ]
            lowered = text.lower()
            if any(f in lowered for f in forbidden):
                logger.warning("Blocked sensitive output content.")
                return "Request not permitted."
            return text
        # Start execution trace
        self._emit_event(
            "task_start",
            {"task": task, "context_keys": list(context.keys()) if context else []},
        )

        df = context.get("df_clean") if context else None
        if df is None or not isinstance(df, pd.DataFrame):
            self._emit_event(
                "task_error",
                {"error": "Invalid or missing DataFrame", "task": task},
                reward=-1.0,
            )
            return {"answer": f"Executed task: {task}", "preview_rows": []}

        created_by = context.get("created_by", "autogen_system")
        encrypted_filename = context.get("encrypted_filename") or f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        output_format = context.get("output_format")

        output_path, detected_format = None, None

        # --- Output generation based on format ---
        try:
            if output_format:
                fmt = output_format.lower()
                if fmt == "ppt":
                    output_path = generate_ppt(task, df, include_charts=True)
                    detected_format = "ppt"
                elif fmt == "excel":
                    output_path = generate_excel(df, task, include_charts=True)
                    detected_format = "excel"
                elif fmt == "word":
                    output_path = generate_word(df, task, include_charts=True)
                    detected_format = "word"
            else:
                # Fallback: infer from task text
                task_lower = task.lower()
                if "ppt" in task_lower or "presentation" in task_lower:
                    output_path = generate_ppt(task, df, include_charts=True)
                    detected_format = "ppt"
                elif "excel" in task_lower:
                    output_path = generate_excel(df, task, include_charts=True)
                    detected_format = "excel"
                elif "word" in task_lower:
                    output_path = generate_word(df, task, include_charts=True)
                    detected_format = "word"

            # --- Handle fallback case ---
            if not detected_format or not output_path:
                logger.warning("No output format detected; generating direct response.")
                answer = generate_direct_response(task, df)
                preview_rows = [
                    dict(zip(df.columns, row)) for row in df.head(5).values.tolist()
                ]
                return {"answer": answer, "preview_rows": preview_rows}
            
            

            self._emit_event(
                f"{detected_format}_generated",
                {
                    "task": task,
                    "output_path": output_path,
                    "data_shape": df.shape,
                },
                reward=1.0,
            )
        except Exception as e:
            logger.exception(f"Error generating {output_format or 'auto-detected'} output")
            self._emit_event(
                "output_generation_error",
                {"error": str(e), "format": output_format, "task": task},
                reward=-0.5,
            )
            # Fallback on failure
            detected_format = None
            output_path = None

        # --- Upload if file generated ---
        
        # --- Upload if file generated ---
        upload_status = "no upload"
        blob_url = None  

        if output_path and detected_format:
            try:
                api_root = "https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/"
                if not api_root.endswith("/"):
                    api_root += "/"

                file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(
                    detected_format, "dat"
                )
                filename_with_ext = f"{encrypted_filename}.{file_ext}"

                # --- 1️⃣ metadata upload (must use filename_with_ext) ---
                save_url = f"{api_root}PostSavePPTDetailsV2"
                metadata = {
                    "fileName": filename_with_ext,
                    "createdBy": created_by,
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                }
                requests.post(save_url, params=metadata, timeout=20)

                # --- 2️⃣ file upload + Content (MUST use same filename everywhere) ---
                if os.path.exists(output_path):

                    insights, recs = generate_insights(df)
                    answer = generate_direct_response(task, df)

                    insights_str = "\n• ".join(insights) if isinstance(insights, list) else insights
                    recs_str = "\n• ".join(recs) if isinstance(recs, list) else recs

                    data_string = (
                        f"Task: {task}\n\n"
                        f"Insights:\n• {insights_str}\n\n"
                        f"Recommendations:\n• {recs_str}\n\n"
                        f"Answer:\n{answer}"
                    )

                    filtered_obj = {
                        "slide": 1,
                        "title": "Auto-generated Slide",
                        "data": data_string
                    }

                    mime_map = {
                        "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                    mimetype = mime_map.get(detected_format, "application/octet-stream")

                    with open(output_path, "rb") as f:
                        files = {"File": (filename_with_ext, f, mimetype)}

                        data_fields = {
                            "FileName": filename_with_ext,     # ⭐ FIXED
                            "CreatedBy": created_by,           # ⭐ FIXED
                            "Content": json.dumps({"content": [filtered_obj]})
                        }

                        upload_params = {
                            "FileName": filename_with_ext,     # ⭐ MUST MATCH EXACTLY
                            "CreatedBy": created_by
                        }

                        upload_url = f"{api_root}UpdatePptFileV2"
                        upload_response = requests.post(
                            upload_url,
                            data=data_fields,
                            files=files,
                            params=upload_params,
                            timeout=60,
                        )

                        if upload_response.status_code == 200:
                            blob_url = f"https://iscmadls.blob.core.windows.net/supplysense-presentations/generated-files-v2/{filename_with_ext}"
                            upload_status = f"{detected_format.upper()} uploaded successfully"
                        else:
                            upload_status = f"Upload failed: {upload_response.status_code}"
                else:
                    upload_status = f"Upload failed: file not found at {output_path}"

            except Exception as e:
                upload_status = f"Upload error: {str(e)}"

                logger.exception("execute: Exception during file upload")

        # --- Construct final result safely ---
        preview_rows = [dict(zip(df.columns, row)) for row in df.head(5).values.tolist()]
        result = {}

        if detected_format and output_path and blob_url:
            #filename_with_ext = os.path.basename(output_path) # Get the filename (e.g., 'report_....pptx')
            filename_with_ext = os.path.basename(urlparse(blob_url).path)
            result["answer"] = f"Generated {detected_format.upper()}: [{filename_with_ext}]({blob_url})"
        else:
            result["answer"] = f"⚠️ Output generation failed for task: {task}"
            detected_format = detected_format or "unknown"
            output_path = output_path or "N/A"

        result.update({
            "preview_rows": preview_rows,
            "file_type": {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(detected_format, "unknown"),
            "file_path": output_path,
            "upload_status": upload_status,
            "blob_url": blob_url,
             "file_url": blob_url, # 🟢 **FIX 2: Expose the Blob URL for the orchestrator**
            "data": df.to_dict(orient="records"),
            "context_key": self.name.lower().replace(" ", "_"),
        })

        self._emit_event(
            "task_complete",
            {
                "task": task,
                "output_format": detected_format,
                "response_length": len(str(result)),
                "preview_count": len(preview_rows),
            },
            reward=1.0,
        )

        return result

# ==============================
# 🔹 SUPPORT CLASSES & HELPERS
# ==============================
class MessageBus:
    def publish(self, topic: str, message: dict):
        """Placeholder message bus for events."""
        pass

_client = client

# ==============================
# 🔹 AGENT CREATION CONVERSATION (unchanged)
# ==============================
async def guide_agent_creation_conversation():
    conversation = []
    agent_name = await ask_user("What would you like to name your agent?", conversation)

    while True:
        role = await ask_user(
            f"What is the role of your '{agent_name}'? (e.g., Inventory Planner, Forecast Analyst)",
            conversation,
        )
        if role not in VALID_ROLES:
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"'{role}' is not a valid role. Choose from: {', '.join(VALID_ROLES)}",
                }
            )
        else:
            break

    purpose = await ask_user(
        f"What is the main purpose of '{agent_name}'? (e.g., Analyze inventory, Forecast demand)",
        conversation,
    )

    while True:
        instructions = await ask_user(
            f"What instructions should '{agent_name}' follow?", conversation
        )
        result = validate_purpose_and_instructions(role, purpose, [instructions])
        if result["purpose_valid"] and not result["invalid_instructions"]:
            break
        err = []
        if not result["purpose_valid"]:
            err.append("❌ Purpose is invalid.")
        if result["invalid_instructions"]:
            err.append(
                f"❌ Invalid instructions: {result['invalid_instructions']}"
            )
        conversation.append(
            {"role": "assistant", "content": " ".join(err) + " Please try again."}
        )

    prompts = generate_sample_prompts(purpose)

    return {
        "name": agent_name,
        "role": role,
        "purpose": purpose,
        "instructions": [instructions],
        "sample_prompts": prompts,
    }

async def ask_user(prompt, conversation_history):
    conversation_history.append({"role": "assistant", "content": prompt})
    response = input(f"{prompt}\n> ")
    conversation_history.append({"role": "user", "content": response})
    return response

VALID_CAPABILITIES = {
    "Summarize results",
    "Generate output as PPT",
    "Generate Excel output",
    "Send email reports",
    "Highlight exceptions",
    "Forecast metrics",
    "Recommend actions",
    "Explain trends",
}

def validate_capabilities(capabilities: list[str]) -> list[str]:
    """Validate given capabilities list."""
    if not isinstance(capabilities, list):
        return ["Capabilities must be a list."]
    invalid = [c for c in capabilities if c not in VALID_CAPABILITIES]
    return invalid