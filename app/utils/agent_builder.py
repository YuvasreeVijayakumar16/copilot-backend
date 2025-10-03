# === ✅ Validate if the agent role is supported
VALID_ROLES = [
    "Supply Chain Planner",
    "Demand Planner",
    "Production Planner",
    "Supply Chain Analyst",
    "Sales & Operations Planning (S&OP) Manager",
    "Inventory Planner",
    "Capacity Planning Specialist",
    "Procurement Specialist / Buyer",
    "Strategic Sourcing Manager",
    "Category Manager",
    "Supplier Relationship Manager",
    "Contract & Compliance Manager",
    "Vendor Development Executive",
    "Global Sourcing Specialist",
    "Supply Chain Data Analyst",
    "ERP/SAP Supply Chain Consultant",
    "Forecasting Analyst",
    "AI/ML Supply Chain Modeler",
    "Digital Supply Chain Transformation Manager",
    "Inventory Optimization Specialist"
]

from openai import OpenAI
import os

_openai_client = None

def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def validate_agent_role(role: str, purpose: str) -> bool:
    """Ensure the user-selected role is from the valid list."""
    return role in VALID_ROLES


# === ✅ Generate sample prompts based on purpose or role
# def generate_sample_prompts(role: str, purpose: str) -> list[str]:
#     purpose_lower = purpose.lower()

#     if "forecast" in purpose_lower:
#         return [
#             "What is the expected forecat for next month?",
#             "Give me a monthly forecast summary",
#             "Create a presentation of next month’s forecast"
#         ]
#     elif "inventory" in purpose_lower:
#         return [
#             "List SKUs with excess inventory",
#             "Show me inventory turnover vs demand",
#             "Which items have slow-moving inventory?"
#         ]
#     elif "supplier" in purpose_lower or "vendor" in purpose_lower:
#         return [
#             "Which suppliers are most delayed?",
#             "Show supplier performance scorecard",
#             "List top 5 vendors by leadtime and PO"
#         ]
#     elif "procure" in purpose_lower or "sourcing" in purpose_lower:
#         return [
#             "Give me a report of high-value purchases last month",
#             "List items with highest sourcing lead time",
#             "Create a ppt of sourcing cost breakdown"
#         ]
#     elif "capacity" in purpose_lower:
#         return [
#             "Which plants are running at full capacity?",
#             "What is my capacity utilization by week?",
#             "Create a report of idle resources"
#         ]
#     else:
#         return [
#             f"What are my insights for {purpose_lower}?",
#             f"Generate a ppt for {purpose_lower}",
#             f"Summarize key metrics for {purpose_lower}"
#         ]


def generate_sample_prompts(purpose: str, role: str = None) -> list[str]:
    """
    Generate dynamic prompts based on the given purpose and optional role using OpenAI.
    """
    role_text = f" for a {role}" if role else ""
    client = _get_client()
    response = client.chat.completions.create(
        model=os.getenv("AUTOGEN_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are an assistant that generates concise, role-aware prompts."},
            {"role": "user", "content": f"Generate 3 example prompts about '{purpose}'{role_text}."}
        ],
        max_tokens=150,
        temperature=0.7
    )
 
    content = response.choices[0].message.content.strip()
    prompts = [
        line.strip("-•1234567890. ").strip()
        for line in content.split("\n")
        if line.strip()
    ]
    return prompts[:3]