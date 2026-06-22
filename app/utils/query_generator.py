# app/utils/query_generator.py
#
# FIX: Now uses the shared singleton client from openai_client.py instead of
# creating its own raw AzureOpenAI instance with a stale api_version.

from app.utils.openai_client import get_openai_client, get_openai_model


def generate_sql_with_openai(question: str, schema: dict, system_prompt: str) -> str:
    """
    Generate a SQL SELECT query from a natural-language question and a schema dict.

    Args:
        question:      The user's question in plain English.
        schema:        Dict mapping table names to lists of column names.
        system_prompt: The system instructions to pass to the model.

    Returns:
        A clean SQL query string (no markdown fences).

    Raises:
        ValueError: If the model returns no content (e.g. content-filter block).
    """
    client = get_openai_client()
    model = get_openai_model()

    schema_text = "\n".join(
        [f"{table}: {', '.join(columns)}" for table, columns in schema.items()]
    )

    prompt = (
        f"### Schema ###\n"
        f"{schema_text}\n\n"
        f"### Question ###\n"
        f"{question}\n\n"
        f"Write an SQL Server SELECT query to answer the question based on the schema.\n"
        f"Return only the SQL query without explanation or markdown."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    # ✅ FIX: Guard against content_filter / None content
    choice = response.choices[0] if response and response.choices else None
    if choice is None:
        raise ValueError("OpenAI returned an empty response (no choices).")

    finish_reason = getattr(choice, "finish_reason", "unknown")
    content = getattr(choice.message, "content", None)

    if content is None:
        if finish_reason == "content_filter":
            raise ValueError(
                "Azure OpenAI content filter blocked the SQL-generation prompt. "
                "Try rephrasing the question."
            )
        raise ValueError(
            f"OpenAI message content is None (finish_reason='{finish_reason}'). "
            "Cannot generate SQL."
        )

    return content.strip().strip("```sql").strip("```").strip()
