import json
import re
import os
from openai import AzureOpenAI, OpenAIError
from app.utils.openai_client import get_openai_model

client = AzureOpenAI(
    api_key=os.getenv("CLOUD_PROVIDER_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def validate_purpose_and_instructions(purpose, instructions, structured_schema, sample_data):
    system_validator_prompt = """You are an AI assistant designed to validate user requests for a database query agent. Your sole purpose is to determine if a given 'purpose' and 'instructions' are valid and feasible based on a provided database schema and sample data.
You must adhere to the following rules for validation:
1.  **Purpose Validity**: A purpose is valid if it clearly relates to **querying, retrieving, analyzing, summarizing, reporting, or deriving insights from the provided database data**. It must directly aim to understand or extract information from the database.
2.  **Instruction Validity**:
    * An instruction is **invalid** if it asks for actions that cannot be performed by querying a database (e.g., modifying data like INSERT/UPDATE/DELETE/DROP, performing external actions like sending emails, making subjective opinions, or asking for facts not present in the database).
    * An instruction is **invalid** if it is too vague or generic (e.g., "do something useful," "tell me a story," "cook a recipe").
    * An instruction is **valid** if it clearly specifies how to filter, aggregate, or present data that *can* be retrieved from the database.
3.  **Strict Output Format**: Your response MUST be a JSON object with two keys: "purpose_valid" (boolean) and "invalid_instructions" (an array of strings). Do NOT include any additional text, markdown formatting (like ```json), or explanations outside of the JSON.
 You must only return JSON like this:
{
  "purpose_valid": true,
  "invalid_instructions": ["cook the recipe", "sing a song"]
}
"""

    # Preview only a few sample records
    sample_data_preview = list(sample_data)[:5]

    # User message prompt
    prompt = f"""
You are validating if a user-defined agent purpose and instruction are aligned with the database shown below.

### STRUCTURED SCHEMA ###
{json.dumps(structured_schema, indent=2)}

### SAMPLE DATA ###
{json.dumps(sample_data_preview, indent=2)}

Evaluate the following:

Purpose:
{purpose}

Instructions:
{instructions}

Return JSON:
{{
  "purpose_valid": true or false,
  "invalid_instructions": ["..."]
}}
"""

    try:
        # Send request to OpenAI
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": system_validator_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # Clean JSON if surrounded by ```json
        cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "purpose_valid": False,
                "invalid_instructions": [],
                "error": f"JSON parsing failed: {e}",
                "raw": cleaned
            }

        return {
            "success": True,
            "purpose_valid": result.get("purpose_valid", False),
            "invalid_instructions": result.get("invalid_instructions", []),
            "error": None,
            "raw": cleaned
        }

    except OpenAIError as e:
        return {
            "success": False,
            "purpose_valid": False,
            "invalid_instructions": [],
            "error": str(e),
            "raw": None
        }
