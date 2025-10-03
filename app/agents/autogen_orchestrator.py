import os
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
import re
import json
import requests
from datetime import datetime
import uuid 
from app.services.agent_servies import (
    load_agent_config,
    is_question_supported_by_capabilities,
    detect_output_format,
)
    
from app.utils.schema_reader import get_schema_and_sample_data
from app.services.agent_servies import (
    validate_question_safety,
    validate_ethical_use,
    is_sql_read_only,
    enforce_sql_row_limit,
    validate_sql_tables,
)
from app.db.sql_connection import execute_sql_query
from app.utils.ppt_generator import (
    generate_ppt,
    generate_excel,
    generate_word,
    generate_insights,
    generate_direct_response,
)
import numpy as np

logger = logging.getLogger("app.agents.autogen_orchestrator")


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _get_model() -> str:
    return os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")

def _extract_select_query(raw_text: str) -> str:
    """Extract the first read-only SELECT statement from LLM output.
    - Strips code fences and commentary
    - Takes content starting at the first 'select' (case-insensitive)
    - Stops at the first semicolon if present
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    # Remove code fences
    if text.startswith("```"):
        text = text.strip("`\n").split("\n", 1)[-1]
    # Find first occurrence of 'select'
    lowered = text.lower()
    idx = lowered.find("select")
    if idx == -1:
        return ""
    text = text[idx:]
    # Cut at first semicolon if any
    semi = text.find(";")
    if semi != -1:
        text = text[:semi]
    # Single line cleanup
    return " ".join(text.split())

def _strip_non_tsql_limits(sql_text: str) -> str:
    """Remove non-T-SQL limit syntaxes like LIMIT/FETCH FIRST to avoid conflicts with TOP."""
    if not sql_text:
        return sql_text
    cleaned = sql_text
    # Remove MySQL/Postgres LIMIT n
    cleaned = re.sub(r"(?i)\s+limit\s+\d+\s*$", "", cleaned)
    # Remove FETCH FIRST n ROW ONLY (DB2/Oracle style)
    cleaned = re.sub(r"(?i)\s+fetch\s+first\s+\d+\s+rows?\s+only\s*$", "", cleaned)
    # Remove OFFSET ... ROWS [FETCH NEXT n ROWS ONLY]
    cleaned = re.sub(r"(?i)\s+offset\s+\d+\s+rows(\s+fetch\s+next\s+\d+\s+rows\s+only)?\s*$", "", cleaned)
    # Remove trailing semicolon
    cleaned = re.sub(r";\s*$", "", cleaned)
    return cleaned

def _strip_markdown_link(text: str) -> str:
    """Strips a string accidentally wrapped in markdown link format [text](url)."""
    # Pattern: Matches [any text](any url)
    # We want to extract the URL inside the parentheses
    match = re.search(r'^\[.*?\]\((.*?)\)$', text.strip())
    if match:
        # Return the URL part (group 1)
        return match.group(1).strip()
    return text.strip()


def run_autogen_orchestration(
    question: str,
    agent_name: Optional[str] = None,
    created_by: Optional[str] = None,
    encrypted_filename: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    previous_results: Optional[List[Dict]] = None,
    output_format: Optional[str] = None,
) -> Dict[str, Any]:
    
    # ----------------------------------------------------
    # 🟢 Define and Sanitize Safe Internal Variables (Base Name Only)
    # ----------------------------------------------------
    
    # 1. Sanitize CreatedBy: Ensure a non-empty, unique-enough ID if not provided.
    _created_by = (created_by if created_by else "System_Orchestrator").strip() 
    if not _created_by:
        # Fallback to a unique ID to satisfy strict backend validation
        _created_by = f"System_Fallback_User_{uuid.uuid4().hex[:8]}" 

    # 2. Sanitize EncryptedFilename (Base Name): Create a robust name from question if none provided.
    if encrypted_filename:
        # Safely get the base filename, stripping the extension and whitespace
        base_name = os.path.splitext(encrypted_filename)[0].strip()
        # If the input name resulted in an empty string, generate a default one
        _encrypted_filename = base_name if base_name else f"Auto_Default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        # Generate robust filename from question and timestamp
        q_part = question[:30] if question else "query"
        # Only keep alphanumeric characters from the question part, replacing others with '_'
        sanitized_q = "".join(c if c.isalnum() else '_' for c in q_part).strip('_')
        if not sanitized_q:
            sanitized_q = "query" # Final safeguard for question part
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _encrypted_filename = f"Auto_{sanitized_q}_{timestamp}" 

    # 3. Final Fallback Check (CRUCIAL): Ensure absolutely non-empty before use
    if not _encrypted_filename:
        _encrypted_filename = f"Fallback_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    logger.info("Sanitized variables ready for upload", extra={"upload_filename": _encrypted_filename, "created_by": _created_by})

    # ----------------------------------------------------
    
    # Step 1: Get schema
    structured_schema, _, _ = get_schema_and_sample_data()
    logger.info("Structured schema loaded", extra={"tables": list(structured_schema.keys())})
    if not structured_schema:
        return {"error": "No database schema available"}

    # Step 2: OpenAI setup
    client = _get_openai_client()
    model = _get_model()

    # Step 3: Add context
    context_str = ""
    if context:
        context_str = "\n\nContext from previous agents:\n"
        for key, value in context.items():
            context_str += f"{key}: {str(value)[:200]}...\n"

    if previous_results:
        context_str += "\n\nPrevious agent results:\n"
        for result in previous_results:
            agent_name = result.get("agent", "Unknown")

            def safe_truncate(text, max_length):
                if len(text) <= max_length:
                    return text
                truncated = text[:max_length]
                if " " in truncated:
                    truncated = truncated.rsplit(" ", 1)[0]
                return truncated

            answer = safe_truncate(result.get("answer", ""), 200)
            context_str += f"{agent_name}: {answer}...\n"

    # Step 4: Planning prompt
    plan_prompt = (
        "You are a planning assistant. Decide if a file (ppt/excel/word) is needed and "
        "draft a clear, concise plan for getting the data and presenting it in the specified format."
        f"\nUser question: {question}"
        f"{context_str}"
        f"\nAllowed tables: {list(structured_schema.keys())}"
    )

    try:
        plan_resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful planning assistant."},
                {"role": "user", "content": plan_prompt},
            ],
        )
        plan_text = (plan_resp.choices[0].message.content or "").strip()
        logger.info("Plan generated by GPT", extra={"plan": plan_text})
    except Exception as e:
        logger.exception("Failed to generate plan", extra={"question": question})
        return {"error": f"Planning failed: {str(e)}"}

    # Step 5: SQL generation
    sql_prompt = (
        "Generate exactly one SQL Server SELECT query only.\n"
        "Requirements:\n"
        "- Read-only (SELECT only).\n"
        "- No comments, no explanation, no CTE, no DDL/DML.\n"
        "- Do NOT include a trailing semicolon.\n"
        "- Use only these tables/columns: " + str(structured_schema) + "\n"
        f"{context_str}"
        "Task: " + question
    )

    try:
        sql_resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You generate a single SQL Server SELECT query only."},
                {"role": "user", "content": sql_prompt},
            ],
        )
        if not sql_resp.choices:
            return {"error": "No SQL generated by LLM"}

        sql_text = (sql_resp.choices[0].message.content or "").strip()
        sql_text = _extract_select_query(sql_text)
        sql_text = _strip_non_tsql_limits(sql_text)

        if not is_sql_read_only(sql_text):
            return {"error": "Generated SQL not read-only"}

        sql_text = enforce_sql_row_limit(sql_text)
        ok_tables, bad = validate_sql_tables(sql_text, list(structured_schema.keys()))
        if not ok_tables:
            return {"error": "Unauthorized table access", "tables": bad}

    except Exception as e:
        logger.exception("SQL generation failed", extra={"question": question})
        return {"error": f"SQL generation failed: {str(e)}"}

    # Step 6: Execute SQL
    df = execute_sql_query(sql_text)
    if df is None or df.empty:
        return {"error": "No data returned from SQL query"}

    # Clean data
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].fillna("null")

    # Step 7: Generate answer
    answer = generate_direct_response(question, df_clean)
    insights, recs = generate_insights(df_clean)

    # Step 8: Generate file (if agent has capabilities)
    file_path = None
    file_type = None
    if agent_name:
        agent_cfg = load_agent_config(agent_name)
        if agent_cfg:
            # Allow caller to override format detection for testing/explicit control
            # The file type is detected based on the question and the generated plan text.
            fmt = output_format if output_format else detect_output_format((question or "") + " " + (plan_text or ""))
            logger.info("Orchestrator: output format decided", extra={"fmt": fmt, "agent": agent_name})

            # Tolerant capability matching: normalize capability strings and support synonyms
            required_capability = {
                "ppt": "generate output as ppt",
                "excel": "generate output as excel",
                "word": "generate output as word",
            }

            def _normalize(s: str) -> str:
                return (s or "").strip().lower()

            capabilities = [
                _normalize(c) for c in getattr(agent_cfg, "capabilities", []) if isinstance(c, str)
            ]
            logger.info("Agent capabilities (normalized)", extra={"caps": capabilities})

            def _capability_supports_format(caps: List[str], fmt_key: str) -> bool:
                if not fmt_key or fmt_key == "none":
                    return False
                # direct required capability match
                req = required_capability.get(fmt_key, "")
                for c in caps:
                    if req and req in c:
                        return True
                # fallback: check if capability mentions the format keyword
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

            if _capability_supports_format(capabilities, fmt):
                include_charts = any(k in question.lower() for k in ["chart", "graph", "visual", "visualize"])
                try:
                    if fmt == "ppt":
                        file_path = generate_ppt(question, df_clean, include_charts=include_charts)
                        file_type = "ppt"
                    elif fmt == "excel":
                        file_path = generate_excel(df_clean, question, include_charts=include_charts)
                        file_type = "excel"
                    elif fmt == "word":
                        file_path = generate_word(df_clean, question, include_charts=include_charts)
                        file_type = "word"

                    # *** CRITICAL CHECK: Ensure Absolute Path ***
                    if file_path:
                        file_path = os.path.abspath(file_path)
                        logger.info(f"File successfully generated. Absolute path for upload: {file_path}")
                    # **********************************************

                except Exception as e:
                    # Log a detailed exception if file generation failed
                    logger.exception(f"File generation failed for format '{fmt}': {e}")
                    # IMPORTANT: Set file_path back to None so upload is skipped
                    file_path = None 
            else:
                logger.info("Agent does not declare capability for requested output format", extra={"fmt": fmt})


    # Step 9: Upload file (optional)
    result: Dict[str, Any] = {
        "plan": plan_text,
        "sql": sql_text,
        "preview_rows": df_clean.head(10).to_dict(orient="records"),
        "answer": answer,
        "insights": insights,
        "recommendations": recs,
    }

    # 🟢 Check the SANITIZED variables for existence before attempting upload
    if file_path and file_type and _created_by and _encrypted_filename:
        
        # 🟢 Define file extension and full filename (with extension)
        file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(file_type, "dat")
        filename_with_ext = f"{_encrypted_filename}.{file_ext}" # e.g. "Auto_query_20251002_150000.pptx"

        # Add file path to result for visibility
        result[f"{file_type}_path"] = file_path
        
        # 🟢 Use the SANITIZED variables from the start of the function
        api_root_raw = "[https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/](https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/)"
        # 🟢 FIX: Clean the API root string to remove any Markdown link contamination
        api_root = _strip_markdown_link(api_root_raw)
        
        # Ensure it has a trailing slash for correct concatenation
        if not api_root.endswith('/'):
            api_root += '/'

        # 1) POST metadata as form data to PostSavePPTDetailsV2
        try:
            save_url = f"{api_root}PostSavePPTDetailsV2"

            # 🟢 Use the full filename WITH extension in the metadata for record creation
            metadata = {
                "fileName": str(filename_with_ext),
                "createdBy": str(_created_by),
                "Date": datetime.now().strftime('%Y-%m-%d'),
            }

            logger.info("Uploading metadata to PostSavePPTDetailsV2", extra={"url": save_url, "metadata": metadata})
            save_resp = requests.post(save_url, params=metadata, timeout=20)
            result["metadata_status"] = save_resp.status_code
            result["metadata_response"] = save_resp.text[:1000]
            if save_resp.status_code != 200:
                result.setdefault("upload_warnings", []).append(f"Metadata save failed: {save_resp.status_code}")
                # FIX: Corrected indentation for extra dictionary
                logger.error("Metadata save failed", extra={
                    "status": save_resp.status_code, 
                    "response_text": save_resp.text[:500] 
                })
        except Exception as e:
            logger.exception("Metadata POST failed")
            result["metadata_error"] = str(e)


        # 2) POST file bytes with multipart form to UpdatePptFileV2
        try:
            # *** NEW: Define filtered_obj here for upload payload ***
            filtered_obj = {"slide": 1, "title": "Auto-generated Slide", "data": question}
            
            # *** CRITICAL CHECK: file_path is now guaranteed to be absolute from Step 8 ***
            if not os.path.exists(file_path):
                # This message indicates a definitive failure of file creation
                result["upload_status"] = f"Upload failed: generated file NOT FOUND on disk at {file_path}. Check file generation logs for Step 8 failure."
                logger.error("Generated file missing (os.path.exists failed)", extra={"file_path": file_path})
            else:
                mime_map = {
                    "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                }
                mimetype = mime_map.get(file_type, "application/octet-stream")

                with open(file_path, "rb") as f:
                    # The actual file content: tuple (filename on server, file object, mimetype)
                    #files = {"file": (filename_with_ext, f, mimetype)}
                    files = {
                       "file": (filename_with_ext, f, mimetype)
                            }


                    _encrypted_filename = filename_with_ext
                    # Use the sanitized _created_by for consistency
                    

                    
                    # 🟢 CRITICAL FIX: Use the BASE filename (no extension) in the 'data' payload
                    # This often serves as the unique key/ID that the server expects to find
                    # from the initial metadata POST.
                    data = {
    "content": json.dumps({"content": [filtered_obj]}),
}
                    upload_params = {
    "FileName": _encrypted_filename,
    "CreatedBy": _created_by,
}
                    upload_url = f"{api_root}UpdatePptFileV2"

                    upload_resp = requests.post(
                        upload_url,
                        data=data,
                        files=files,
                        params=upload_params,
                        timeout=60,
                    )


                    

                    result["upload_code"] = upload_resp.status_code
                    result["upload_response"] = upload_resp.text[:2000]
                    if upload_resp.status_code == 200:
                        result["upload_status"] = f"{file_type.upper()} uploaded successfully"
                    else:
                        result["upload_status"] = f"Upload failed: {upload_resp.status_code}"
                        # FIX: Corrected indentation for extra dictionary
                        logger.warning("File upload failed", extra={
                            "status": upload_resp.status_code, 
                            "text": upload_resp.text[:500]
                        })
        except Exception as e:
            logger.exception(f"Exception during file upload for '{file_type}'")
            result["upload_status"] = f"Upload error: {str(e)}"
            
        # ----------------------------------------------------
        # 🟢 Confirmation that the local copy is kept (as requested)
        # ----------------------------------------------------
        if os.path.exists(file_path):
            logger.info("Local file saved successfully and is being preserved as requested.", extra={"preserved_path": file_path})
        else:
            logger.warning("Local file cleanup issue: file was not found after attempted upload.", extra={"expected_path": file_path})


    return result
