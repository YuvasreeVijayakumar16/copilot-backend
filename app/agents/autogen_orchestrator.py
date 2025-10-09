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
import pandas as pd

logger = logging.getLogger("app.agents.autogen_orchestrator")


def _get_openai_client() -> OpenAI:
    """Retrieve the OpenAI client instance with a cached approach to minimize latency."""
    global _client_instance
    if '_client_instance' not in globals():
        _client_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client_instance


def _get_model() -> str:
    """Retrieve the model name from environment variable with a default fallback."""
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
    
    # Step 1: Get schema (cache the schema for reuse)
    if not hasattr(run_autogen_orchestration, "_cached_schema"):
        structured_schema, _, _ = get_schema_and_sample_data()
        run_autogen_orchestration._cached_schema = structured_schema
    else:
        structured_schema = run_autogen_orchestration._cached_schema
    
    logger.info("Using cached structured schema", extra={"tables": list(structured_schema.keys())})
    if not structured_schema:
        return {"error": "No database schema available"}

    client = _get_openai_client()  # Ensures a single instantiation of client for efficiency
    model = _get_model()

    # Load agent config if agent_name provided
    agent_config = None
    if agent_name:
        agent_config = load_agent_config(agent_name)
        if not agent_config:
            return {"error": f"Agent '{agent_name}' not found"}

    # Set output_format from agent config if available
    if not output_format and agent_config and hasattr(agent_config, 'output_method') and agent_config.output_method != "chat":
        output_format = agent_config.output_method

    # Generate SQL query
    schema_text = "\n".join([f"{table}: {', '.join(cols)}" for table, cols in structured_schema.items()])
    prompt = f"Generate a SQL query to answer the following question: '{question}'. Use the exact table names and column names from the database schema provided below. Do not assume or invent table or column names; use only those listed in the schema. Use TOP 100 to limit results. Return only the SQL query, no explanations or markdown.\n\nDatabase schema (table: columns):\n{schema_text}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    sql_query = response.choices[0].message.content.strip()
    # Clean up the query
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    logger.info("Generated SQL query", extra={"query": sql_query})

    # Execute the query
    result = execute_sql_query(sql_query)
    if isinstance(result, dict) and "error" in result:
        logger.error("SQL query failed", extra={"error": result["error"]})
        return {"error": result["error"]}

    df = pd.DataFrame(result)
    if df.empty:
        return {"answer": "No data found for the query."}

    # Clean the dataframe
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].fillna("null")

    # Detect output format from question if not provided
    if not output_format:
        question_lower = question.lower()
        if "ppt" in question_lower or "powerpoint" in question_lower:
            output_format = "ppt"
        elif "excel" in question_lower:
            output_format = "excel"
        elif "word" in question_lower or "doc" in question_lower:
            output_format = "word"

    # Generate the direct response as the answer
    answer = generate_direct_response(question, df)

    # Generate insights and recommendations
    insights, recs = generate_insights(df)

    # Generate additional file if output_format specified
    file_path = None
    file_type = None
    if output_format == "ppt":
        file_path = generate_ppt(question, df)
        file_type = "ppt"
    elif output_format == "excel":
        file_path = generate_excel(df, question)
        file_type = "excel"
    elif output_format == "word":
        file_path = generate_word(df, question)
        file_type = "word"

    result = {
        "plan": "Optimized orchestration: Direct SQL generation and answer without planning LLM call for speed.",
        "sql": sql_query,
        "data": df_clean.to_dict(orient="records"),
        "answer": answer,
        "insights": insights,
        "recommendations": recs,
    }
    if file_path and file_type and created_by and encrypted_filename:
        # Define file extension and full filename (with extension)
        file_ext = {"ppt": "pptx", "excel": "xlsx", "word": "docx"}.get(file_type, "dat")
        filename_with_ext = f"{encrypted_filename}.{file_ext}"
        # Add file path to result for visibility
        result[f"{file_type}_path"] = file_path
        result["file_type"] = file_type
        result["file_path"] = file_path
        api_root = "https://supplysenseaiapi-aadngxggarc0g6hz.z01.azurefd.net/api/iSCM/"
        if not api_root.endswith('/'):
            api_root += '/'
        # 1) POST metadata as form data to PostSavePPTDetailsV2
        try:
            save_url = f"{api_root}PostSavePPTDetailsV2"
            metadata = {
                "fileName": str(filename_with_ext),
                "createdBy": str(created_by),
                "Date": datetime.now().strftime('%Y-%m-%d'),
            }
            logger.info("Uploading metadata to PostSavePPTDetailsV2", extra={"url": save_url, "metadata": metadata})
            save_resp = requests.post(save_url, params=metadata, timeout=20)
            result["metadata_status"] = save_resp.status_code
            result["metadata_response"] = save_resp.text[:1000]
            if save_resp.status_code != 200:
                result.setdefault("upload_warnings", []).append(f"Metadata save failed: {save_resp.status_code}")
                logger.error("Metadata save failed", extra={"status": save_resp.status_code, "response_text": save_resp.text[:500]})
        except Exception as e:
            logger.exception("Metadata POST failed")
            result["metadata_error"] = str(e)
        # 2) POST file bytes with multipart form to UpdatePptFileV2
        try:
            if not os.path.exists(file_path):
                result["upload_status"] = f"Upload failed: generated file not found at {file_path}"
                logger.error("Generated file missing", extra={"file_path": file_path})
            else:
                mime_map = {
                    "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                }
                mimetype = mime_map.get(file_type, "application/octet-stream")
                filtered_obj = {"slide": 1, "title": "Auto-generated Slide", "data": question} if file_type == "ppt" else {"sheet": 1, "title": "Auto-generated Sheet", "data": question} if file_type == "excel" else {"paragraph": 1, "title": "Auto-generated Paragraph", "data": question}
                with open(file_path, "rb") as f:
                    files = {"file": (filename_with_ext, f, mimetype)}
                    data = {"content": json.dumps({"content": [filtered_obj]})}
                    upload_params = {"FileName": filename_with_ext, "CreatedBy": created_by}
                    upload_url = f"{api_root}UpdatePptFileV2"
                    upload_resp = requests.post(upload_url, data=data, files=files, params=upload_params, timeout=60)
                    result["upload_code"] = upload_resp.status_code
                    result["upload_response"] = upload_resp.text[:2000]
                    if upload_resp.status_code == 200:
                        result["upload_status"] = f"{file_type.upper()} uploaded successfully"
                    else:
                        result["upload_status"] = f"Upload failed: {upload_resp.status_code}"
                        logger.warning("File upload failed", extra={"status": upload_resp.status_code, "text": upload_resp.text[:500]})
        except Exception as e:
            logger.exception(f"Exception during file upload for '{file_type}'")
            result["upload_status"] = f"Upload error: {str(e)}"
    else:
        if file_path:
            result["file_type"] = file_type
            result["file_path"] = file_path
            result["upload_status"] = "no upload"
    return result
