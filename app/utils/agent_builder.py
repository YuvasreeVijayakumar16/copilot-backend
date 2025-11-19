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
import time
import json

_openai_client = None

def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

# Simple in-memory TTL cache for generated prompts: {(purpose, role): (expiry_ts, prompts)}
_prompts_cache: dict = {}
_PROMPT_CACHE_TTL = int(os.getenv("PROMPT_CACHE_TTL_SECONDS", "3600"))

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


def generate_sample_prompts(purpose: str, role: str | None = None) -> list[str]:
    """
    Generate focused prompts about an agent's specific purpose using OpenAI.
    Backwards-compatible: callers may pass either (purpose) or the legacy (role, purpose).
    Returns three example prompts that demonstrate how to use the agent.
    Uses an intelligent caching system to avoid redundant LLM calls.
    """
    # Backwards compatibility: if caller passed (role, purpose) in that order,
    # detect and normalise. If role is provided but looks like a purpose, it's ok.
    # Accept either: generate_sample_prompts(purpose)
    # or legacy: generate_sample_prompts(role, purpose)
    # Normalize values
    if role is not None and purpose:
        # If first arg looks like a valid role, assume legacy order (role, purpose)
        # In that case 'purpose' actually holds the role value; swap.
        if purpose in VALID_ROLES and role and isinstance(role, str):
            # caller used (role, purpose) -> swap
            role_val = purpose
            purpose_val = role
        else:
            role_val = role
            purpose_val = purpose
    else:
        role_val = role
        purpose_val = purpose

    purpose_text = (purpose_val or "").lower().strip()
    role_text = (role_val or "").strip()
    key = f"role:{role_text}|purpose:{purpose_text}"
    now = int(time.time())
    # Check cache
    cached = _prompts_cache.get(key)
    if cached and cached[0] > now:
        return cached[1]

    client = _get_client()

    system_msg = """You are an expert prompt engineer crafting concise, actionable prompts for AI agents.
Rules:
1. Create exactly 3 prompts that are specific to the agent's purpose
2. Each prompt should focus on a distinct analytical aspect
3. Use clear business metrics and KPIs where relevant
4. Keep prompts under 10 words
5. Start with action verbs like Analyze, Show, Compare, List, etc.
6. Focus on data analysis and insights
Do NOT add any explanations, just return the prompts."""

    user_msg = f"Generate 3 precise, data-focused prompts for a {purpose_val or purpose} agent. The prompts should help extract specific insights."

    try:
        response = client.chat.completions.create(
            model=os.getenv("AUTOGEN_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=120,
            temperature=0.2
        )

        content = (response.choices[0].message.content or "").strip()

        # Try JSON parse first (if the model returned a list)
        prompts = []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                prompts = [str(p).strip() for p in parsed if str(p).strip()]
        except Exception:
            # Fallback: split lines and strip bullets/numbers
            prompts = [
                line.strip("-•\u2022 1234567890.").strip()
                for line in content.split("\n")
                if line.strip()
            ]

        # Final sanitize and take up to 3
        prompts = [p for p in [p.strip() for p in prompts] if p][:3]
        if prompts:
            _prompts_cache[key] = (now + _PROMPT_CACHE_TTL, prompts)
            return prompts

    except Exception:
        # Log & fall through to static fallback
        pass

    # Static fallback (deterministic templates based on purpose keywords)
    purpose_lower = (purpose_val or "").lower()
    def _static_for(purpose_lwr: str):
        if "forecast" in purpose_lwr:
            return [
                "Provide the demand forecast for next month.",
                "Summarize key forecast drivers and assumptions.",
                "Create a short presentation showing weekly forecast trends."
            ]
        if "inventory" in purpose_lwr:
            return [
                "List SKUs with excess inventory and their quantities.",
                "Show inventory turnover and slow-moving items.",
                "Prepare a short PPT summarizing inventory risks."
            ]
        if "supplier" in purpose_lwr or "vendor" in purpose_lwr:
            return [
                "Which suppliers have the highest lead-time variability?",
                "Show supplier on-time performance for the last 6 months.",
                "Create a short slide with top 5 suppliers by delay."
            ]
        if "procure" in purpose_lwr or "sourcing" in purpose_lwr:
            return [
                "List top cost drivers in recent purchase orders.",
                "Which items have the highest sourcing lead time?",
                "Create a PPT slide summarizing sourcing cost breakdown."
            ]
        if "capacity" in purpose_lwr:
            return [
                "Which plants are at or above capacity this month?",
                "Show capacity utilization by site and week.",
                "Prepare a slide with recommendations to address capacity gaps."
            ]
        # Generic fallback
        return [
            f"What are my insights for {purpose_lwr}?",
            f"Generate a PPT summarizing key metrics for {purpose_lwr}.",
            f"Provide 3 recommended actions for {purpose_lwr}."
        ]

    prompts = _static_for(purpose_lower)
    _prompts_cache[key] = (now + _PROMPT_CACHE_TTL, prompts)
    return prompts