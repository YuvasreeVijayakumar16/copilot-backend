# app/utils/query_generator.py

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_sql_with_openai(question, schema, system_prompt):
    schema_text = "\n".join([f"{table}: {', '.join(columns)}" for table, columns in schema.items()])

    prompt = f"""
### Schema ###
{schema_text}

### Question ###
{question}

Write an SQL Server SELECT query to answer the question based on the schema.
Return only the SQL query without explanation or markdown.
"""



    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip().strip("```sql").strip("```")
