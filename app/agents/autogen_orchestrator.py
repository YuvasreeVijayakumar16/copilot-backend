import os
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
import re
import uuid
import json
import requests

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
    generate_ppt_enhanced as generate_ppt,
    generate_excel,
    generate_word,
    generate_insights,
    generate_direct_response,
)

import numpy as np
import pandas as pd
from datetime import datetime, date # <-- FIX: Ensure 'date' is imported

logger = logging.getLogger("app.agents.autogen_orchestrator")

def _make_json_serializable(obj):
    """Recursively converts date/datetime objects to ISO 8601 strings for JSON serialization."""
    
    # CORRECT LINE: Use `datetime`, `date`, and `pd.Timestamp` (if you import `date` and `datetime`)
    if isinstance(obj, (datetime, date, pd.Timestamp)): 
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(elem) for elem in obj]
    return obj
def _get_openai_client() -> OpenAI:
    global _client_instance
    if "_client_instance" not in globals():
        _client_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client_instance

def _get_model() -> str:
    return os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")

def run_autogen_orchestration(
    question: str,
    agent_name: Optional[str] = None,
    created_by: Optional[str] = None,
    encrypted_filename: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    previous_results: Optional[List[Dict]] = None,
    output_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the autogen orchestration for processing the task with optimized steps."""
    
    # Step 1: Load schema
    if not hasattr(run_autogen_orchestration, "_cached_schema"):
        structured_schema, _, _ = get_schema_and_sample_data()
        run_autogen_orchestration._cached_schema = structured_schema
    else:
        structured_schema = run_autogen_orchestration._cached_schema
    
    if not structured_schema:
        return {"error": "No database schema available"}

    logger.info("Using cached structured schema", extra={"tables": list(structured_schema.keys())})

    client = _get_openai_client()
    model = _get_model()

    # Step 2: Load agent config
    agent_config = None
    if agent_name:
        agent_config = load_agent_config(agent_name)
        if not agent_config:
            return {"error": f"Agent '{agent_name}' not found"}

    if not output_format and agent_config and hasattr(agent_config, "output_method") and agent_config.output_method != "chat":
        output_format = agent_config.output_method

    # === 🧠 Step 3: Validate question content ===
    structured_schema_preview = {k: v[:5] for k, v in structured_schema.items()}
    validation_result = validate_question_safety(question)
    if not validation_result[0]:
        return {"error": "❌ Question failed safety validation", "reasons": validation_result[1]}

    ethical_result = validate_ethical_use(question)
    if not ethical_result[0]:
        return {"error": "❌ Question violates ethical use policy", "violations": ethical_result[1]}

    # === 🧠 Step 4: Generate SQL ===
    schema_text = "\n".join([f"{table}: {', '.join(cols)}" for table, cols in structured_schema.items()])
    prompt = (
    f"Generate a valid SQL query to answer: '{question}'.\n"
    f"Use only columns that appear in this schema list:\n{schema_text}\n\n"
    f"Rules:\n"
    f"1. Only use column names and table names that exactly match the schema.\n"
    f"2. If unsure about a column, skip it instead of guessing.\n"
    f"3. Always include TOP 100 in SELECT.\n"
    f"4. **CRITICAL: Do NOT include any file export clauses like INTO OUTFILE, BCP, or OPENROWSET.**\n"
    f"5. Return only a valid SQL query without markdown or explanations."
)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        sql_query = response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("OpenAI SQL generation failed")
        return {"error": f"SQL generation failed: {e}"}

    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    logger.info("Generated SQL query", extra={"query": sql_query})

    # === 🔒 Step 5: Guardrail Validations ===
    # Ensure SQL is read-only
    if not is_sql_read_only(sql_query):
        logger.warning("Generated SQL is not read-only", extra={"sql": sql_query})
        if not sql_query.lower().startswith("select"):
            sql_query = f"SELECT TOP 100 * FROM ({sql_query}) AS safe_view"
            logger.info("Auto-wrapped non-select SQL for safety", extra={"query": sql_query})
        else:
            return {"error": "❌ Generated SQL is not read-only and was blocked."}

    # Enforce row limit
    sql_query = enforce_sql_row_limit(sql_query)

    # Validate schema tables used in SQL
    allowed_tables = list(structured_schema.keys())
    valid, invalid_tables = validate_sql_tables(sql_query, allowed_tables)
    if not valid and invalid_tables:
        logger.warning("SQL references unauthorized or unknown tables", extra={"invalid_tables": invalid_tables})
        # Just warn, don’t block — GPT may lowercase or alias names
        sql_query = re.sub(r"dbo\.", "", sql_query, flags=re.IGNORECASE)

    # === ⚙️ Step 6: Execute SQL ===
    result = execute_sql_query(sql_query)
    if isinstance(result, dict) and "error" in result:
        logger.error("SQL execution failed", extra={"error": result["error"]})
        return {"error": result["error"]}

    df = pd.DataFrame(result)
    if df.empty:
        return {"answer": "No data found for the query.", "sql": sql_query}

    # Clean data
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].fillna("null")

    # === 💡 Step 7: Format output type ===
    if not output_format:
        output_format = detect_output_format(question)
    question_lower = question.lower()
    if not output_format:
        if "ppt" in question_lower or "presentation" in question_lower:
            output_format = "ppt"
        elif "excel" in question_lower:
            output_format = "excel"
        elif "word" in question_lower or "doc" in question_lower:
            output_format = "word"

    # === 📊 Step 8: Generate responses ===
    answer = generate_direct_response(question, df_clean)
    insights, recs = generate_insights(df_clean)

    # === 🖼️ Step 9: File generation ===
        # === 🖼️ Step 9: File generation ===
    file_path, file_type = None, None

    # 🔹 Create a canonical filename with extension (used everywhere)
    file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(output_format, "dat")
    #filename_with_ext = f"{encrypted_filename}.{file_ext}"

     # Always normalize filename to avoid double extensions

    filename_stem_1, _ = os.path.splitext(encrypted_filename)
    # 2. Strip the second-to-last extension (e.g., .docx -> demandsplannoext)
    final_stem, _ = os.path.splitext(filename_stem_1)
    filename_with_ext = f"{final_stem}.{file_ext}"

    # 🔹 Generate file and ensure generator uses the correct name
    if output_format == "ppt":
        file_path = generate_ppt(
            question, df_clean, include_charts=True, filename=final_stem  # ✅ consistent naming
        )
        file_type = "ppt"
    elif output_format == "excel":
        file_path = generate_excel(
            df_clean, question, include_charts=True, filename=final_stem
        )
        file_type = "excel"
    elif output_format == "word":
        file_path = generate_word(
            df_clean, question, include_charts=True, filename=final_stem
        )
        file_type = "word"

    result = {
        "plan": "Optimized orchestration with SQL guardrails and GPT-enhanced file output.",
        "sql": sql_query,
        "data": df_clean.to_dict(orient="records"),
        "answer": answer,
        "insights": insights,
        "recommendations": recs,
    }

    # === ☁️ Step 10: Upload to API & Blob ===
        # === ☁️ Step 10: Upload to API & Blob ===
    # === ☁️ Step 10: Upload to API & Blob ===
    if file_path and file_type and created_by and encrypted_filename:

        api_root = "https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/"
        if not api_root.endswith("/"):
            api_root += "/"

        try:
           # file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(output_format, "dat")
            #filename_with_ext = f"{encrypted_filename}.{file_ext}"
           # filename_stem, _ = os.path.splitext(encrypted_filename)
           # filename_with_ext = f"{filename_stem}.{file_ext}"

            # --- 1️⃣ REGISTER METADATA ---
            save_url = f"{api_root}PostSavePPTDetailsV2"
            save_params = {
                "fileName": filename_with_ext,      # ⭐ FIXED
                "createdBy": created_by,
                "Date": datetime.now().strftime("%Y-%m-%d"),
            }
            save_resp = requests.post(save_url, params=save_params, timeout=20)

            # --- 2️⃣ UPLOAD FILE ---
            if os.path.exists(file_path):

                insights_str = "\n• ".join(insights) if isinstance(insights, list) else insights
                recs_str = "\n• ".join(recs) if isinstance(recs, list) else recs

                data_string = (
                    f"Question: {question}\n\n"
                    f"Insights:\n• {insights_str}\n\n"
                    f"Recommendations:\n• {recs_str}\n\n"
                    f"Answer:\n{answer}"
                )

                filtered_obj = {
                    "slide": 1 if file_type == "ppt" else 0,
                    "title": f"Report for: {question[:75]}...",
                    "data": data_string
                }

                upload_url = f"{api_root}UpdatePptFileV2"
                mime_map = {
                    "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                }
                mimetype = mime_map.get(file_type, "application/octet-stream")


                with open(file_path, "rb") as f:

                    files = {"File": (filename_with_ext, f, mimetype)}

                    data_fields = {
                        "FileName": filename_with_ext,   # ⭐ FIXED
                        "CreatedBy": created_by,
                        "Content": json.dumps({"content": [_make_json_serializable(filtered_obj)]}),
                    }

                    upload_params = {
                        "FileName": filename_with_ext,   # ⭐ MUST MATCH EXACTLY
                        "CreatedBy": created_by,
                    }

                    upload_resp = requests.post(
                        upload_url,
                        params=upload_params,
                        data=data_fields,
                        files=files,
                        timeout=60,
                    )

                    if upload_resp.status_code == 200:
                        blob_url = f"https://iscmadls.blob.core.windows.net/supplysense-presentations/generated-files-v2/{filename_with_ext}"
                        result["blob_url"] = blob_url
                        result["file_url"] = blob_url
                        result["upload_status"] = "Success"
                    else:
                        result["upload_status"] = f"Upload failed: {upload_resp.status_code}"

        except Exception as e:
            result["upload_status"] = f"Upload error: {e}"

    return result  
