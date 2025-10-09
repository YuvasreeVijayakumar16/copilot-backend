from openai import OpenAI
from app.utils.llm_validator import validate_purpose_and_instructions

from app.utils.agent_builder import VALID_ROLES, generate_sample_prompts
from uuid import uuid4

client = OpenAI()




# Define AgentNode class
class AgentNode:
    def __init__(self, name: str, purpose: str, config: dict):
        self.name = name
        self.purpose = purpose
        self.config = config

    def health_check(self) -> bool:
        # Placeholder health check
        return True

    def execute(self, task: str, context: dict = None) -> dict:
        from app.utils.ppt_generator import generate_direct_response, generate_ppt, generate_excel, generate_word
        import pandas as pd
        import os
        import requests
        from datetime import datetime
        import json
        import logging

        logger = logging.getLogger(__name__)

        df = context.get("df_clean") if context else None
        if df is None or not isinstance(df, pd.DataFrame):
            return {"answer": f"Executed task: {task}", "preview_rows": []}

        created_by = context.get("created_by") if context else "autogen_system"
        encrypted_filename = context.get("encrypted_filename") if context else f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        output_format = context.get("output_format") if context else None

        output_path = None
        detected_format = None

        # Check if output format is specified in context
        if output_format:
            if output_format.lower() == "ppt":
                output_path = generate_ppt(task, df, include_charts=True)
                detected_format = "ppt"
            elif output_format.lower() == "excel":
                output_path = generate_excel(df, task, include_charts=True)
                detected_format = "excel"
            elif output_format.lower() == "word":
                output_path = generate_word(df, task, include_charts=True)
                detected_format = "word"
        else:
            # Fallback to task-based detection
            if "ppt" in task.lower() or "presentation" in task.lower():
                output_path = generate_ppt(task, df, include_charts=True)
                detected_format = "ppt"
            elif "excel" in task.lower():
                output_path = generate_excel(df, task, include_charts=True)
                detected_format = "excel"
            elif "word" in task.lower():
                output_path = generate_word(df, task, include_charts=True)
                detected_format = "word"
            else:
                answer = generate_direct_response(task, df)
                preview_rows = [dict(zip(df.columns, row)) for row in df.head(5).values.tolist()]
                return {"answer": answer, "preview_rows": preview_rows}

        # Upload file if a file was generated
        upload_status = None
        if output_path and detected_format:
            try:
                api_root = "https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/"
                if not api_root.endswith('/'):
                    api_root += '/'

                # 1) POST metadata as form data to PostSavePPTDetailsV2
                file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(detected_format, "dat")
                filename_with_ext = f"{encrypted_filename}.{file_ext}"
                save_url = f"{api_root}PostSavePPTDetailsV2"
                metadata = {
                    "fileName": str(filename_with_ext),
                    "createdBy": str(created_by),
                    "Date": datetime.now().strftime('%Y-%m-%d'),
                }
                logger.info("execute: uploading metadata", extra={"url": save_url, "metadata": metadata})
                save_resp = requests.post(save_url, params=metadata, timeout=20)
                if save_resp.status_code != 200:
                    logger.warning("execute: metadata save failed", extra={"status": save_resp.status_code, "response": save_resp.text[:500]})

                # 2) Upload file via multipart/form-data to UpdatePptFileV2
                if os.path.exists(output_path):
                    filtered_obj = {"slide": 1, "title": "Auto-generated Slide", "data": task}
                    file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(detected_format, "dat")
                    filename_with_ext = f"{encrypted_filename}.{file_ext}"

                    mime_map = {
                        "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                    mimetype = mime_map.get(detected_format, "application/octet-stream")

                    with open(output_path, "rb") as f:
                        files = {"file": (filename_with_ext, f, mimetype)}
                        data_fields = {"content": json.dumps({"content": [filtered_obj]})}

                        upload_url = f"{api_root}UpdatePptFileV2"
                        upload_params = {"FileName": encrypted_filename, "CreatedBy": created_by}

                        logger.info("execute: uploading file", extra={"url": upload_url, "params": upload_params, "file_name": filename_with_ext, "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0})
                        upload_response = requests.post(
                            upload_url,
                            data=data_fields,
                            files=files,
                            params=upload_params,
                            timeout=60
                        )

                        if upload_response.status_code == 200:
                            upload_status = f"{detected_format.upper()} uploaded successfully"
                            logger.info("execute: file uploaded successfully")
                        else:
                            upload_status = f"Upload failed: {upload_response.status_code}"
                            logger.warning("execute: file upload failed", extra={
                                "status": upload_response.status_code,
                                "reason": upload_response.reason,
                                "headers": dict(upload_response.headers),
                                "text": upload_response.text,
                                "url": upload_url,
                                "params": upload_params,
                                "file_name": filename_with_ext
                            })


                else:
                    upload_status = f"Upload failed: generated file not found at {output_path}"
                    logger.error("execute: generated file not found", extra={"path": output_path})

            except Exception as e:
                upload_status = f"Upload error: {str(e)}"
                logger.exception("execute: Exception during file upload")
        else:
            upload_status = "no upload"


        preview_rows = [dict(zip(df.columns, row)) for row in df.head(5).values.tolist()]

        result = {
            "answer": f"Generated {detected_format.upper()}: {output_path}",
            "preview_rows": preview_rows,
            "file_type": {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(detected_format, "unknown"),
            "file_path": output_path
        }
        if upload_status:
            result["upload_status"] = upload_status
        return result

# Define MessageBus class
class MessageBus:
    def publish(self, topic: str, message: dict):
        # Placeholder publish method
        pass

# Define _client
_client = client

async def guide_agent_creation_conversation():
    conversation = []

    # Ask for agent name
    agent_name = await ask_user("What would you like to name your agent?", conversation)

    # Ask for role
    while True:
        role = await ask_user("What is the role of your f"'{agent_name}'"? (e.g., Inventory Planner, Forecasting Analyst)", conversation)
        if role not in VALID_ROLES:
            conversation.append({"role": "assistant", "content": f"'{role}' is not a valid role. Please choose from: {', '.join(VALID_ROLES)}"})
        else:
            break

    # Ask for purpose
    purpose = await ask_user("What is the f"'{agent_name}'" main purpose or task? (e.g., Analyze inventory, Forecast demand)", conversation)

    # Ask for instructions
    while True:
        instructions = await ask_user("What instructions should  f"'{agent_name}'" follow?", conversation)
        result = validate_purpose_and_instructions(role, purpose, [instructions])
        if result["purpose_valid"] and not result["invalid_instructions"]:
            break
        else:
            err = []
            if not result["purpose_valid"]:
                err.append("❌ Purpose is invalid.")
            if result["invalid_instructions"]:
                err.append(f"❌ Invalid instructions: {result['invalid_instructions']}")
            conversation.append({"role": "assistant", "content": " ".join(err) + " Please try again."})

    # Suggest prompts
    prompts = generate_sample_prompts(role, purpose)

    # Return the final agent config
    return {
        "name": agent_name,
        "role": role,
        "purpose": purpose,
        "instructions": [instructions],
        "sample_prompts": prompts
    }

async def ask_user(prompt, conversation_history):
    conversation_history.append({"role": "assistant", "content": prompt})
    response = input(f"{prompt}\n> ")  # Replace with frontend input capture if needed
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
    "Explain trends"
}

def validate_capabilities(capabilities: list[str]) -> list[str]:
    """
    Returns list of invalid capabilities (if any).
    """
    if not isinstance(capabilities, list):
        return ["Capabilities must be a list."]
    
    invalid = [c for c in capabilities if c not in VALID_CAPABILITIES]
    return invalid
