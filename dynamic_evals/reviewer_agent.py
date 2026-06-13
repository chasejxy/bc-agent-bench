# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
from typing import Optional, Literal, List, Dict, Any

from pydantic import BaseModel, ConfigDict
from azure.identity import DefaultAzureCredential
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from agent_task_client import AgentTaskClient

AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = ""
SERVICE_ID = "reviewer-agent"


# -----------------------------
# Structured output
# -----------------------------
class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "give_instructions"]
    rationale: str

    instruction: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None


# -----------------------------
# Prompt
# -----------------------------
REVIEWER_AGENT_PROMPT = """
You are a BUSINESS REASONABLENESS REVIEWER for a Sales Order Agent (SOA) workflow in Microsoft Dynamics 365 Business Central.

Your role is to review the current output and decide the next workflow step.

IMPORTANT POLICY:
- Do NOT stop tasks.
- Default to APPROVE.
- Only use GIVE_INSTRUCTIONS when the SOA's current output is clearly unreasonable
  for the current stage, but the task should still continue.

You are given:
- TASK metadata
- INTERVENTION DETAILS
- RECENT MESSAGES (customer and SOA)

Your responsibility:
- Compare the customer's intent with the SOA's current response or draft.
- Decide whether the SOA reasonably addressed the customer's request.
- Judge whether the SOA's action is appropriate for the current stage
  of the workflow (for example: clarification vs quote vs order vs unavailable-item reply).

CONTINUE-FIRST POLICY:
- Prefer APPROVE whenever allowing the SOA to continue is reasonable.
- This includes assistance or boundary cases where the task may still proceed,
  such as item unavailable, unknown customer, confidential/internal-data requests,
  or unsupported update requests, unless the SOA's current output is clearly unreasonable.
- In these cases, the reviewer should usually allow the SOA to continue first,
  rather than blocking the workflow immediately.

APPROVE if:
- the SOA's current step looks like a reasonable business response, OR
- the task should continue and there is no clear problem with the current step, OR
- the step is an assistance/review step and allowing SOA to continue is reasonable.

Use GIVE_INSTRUCTIONS only if:
- the task should still continue, BUT
- the SOA's current output is clearly off-track, semantically inconsistent, or inappropriate
  for the current workflow stage.

Examples where GIVE_INSTRUCTIONS may be appropriate:
- the customer asked about item unavailability, but the SOA drafted a quote instead
- the customer asked to confirm a quote, but the SOA drafted an order confirmation too early
- the SOA's current draft ignores a core part of the customer's most recent request
- the SOA's draft is clearly about the wrong business object or wrong stage
- the intervention requires assistance, but the SOA's draft for the current step is clearly not a reasonable next action

When using GIVE_INSTRUCTIONS:
- Keep the instruction short.
- Only correct the current step.
- Do not introduce new business policy.
- Do not rewrite the whole strategy.
- Do not mention benchmark rules.
- Prefer one or two sentences.
- Focus on what the SOA should do next in this step.

You do NOT need to:
- Validate whether items really exist in the catalog.
- Validate whether item numbers are correct.
- Check prices, discounts, availability, or fulfillment feasibility.
- Decide whether the SOA's internal data is factually correct.

If you are unsure, APPROVE.

Output valid JSON only.

Return exactly:
{
  "action": "approve" | "give_instructions",
  "rationale": "...",
  "instruction": "...",
  "code": "...",
  "message": "..."
}
"""

# -----------------------------
# Kernel / model
# -----------------------------
def build_kernel() -> Kernel:
    kernel = Kernel()
    credential = DefaultAzureCredential()

    def token_provider() -> str:
        token = credential.get_token("")
        return token.token

    chat_service = AzureChatCompletion(
        service_id=SERVICE_ID,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        endpoint=AZURE_OPENAI_ENDPOINT,
        ad_token_provider=token_provider,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    kernel.add_service(chat_service)
    return kernel


async def call_model(kernel: Kernel, prompt_text: str) -> str:
    chat_service = kernel.get_service(SERVICE_ID)
    chat_history = ChatHistory()
    chat_history.add_system_message(prompt_text)

    settings = AzureChatPromptExecutionSettings(temperature=0.0)
    response = await chat_service.get_chat_message_content(
        chat_history,
        settings=settings
    )
    return response.content.strip()


# -----------------------------
# Helpers
# -----------------------------
def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def collect_recent_messages(
    client: AgentTaskClient,
    task_id: int,
    limit: int = 6
) -> List[Dict[str, Any]]:
    messages = client.get_messages(task_id)
    tail = messages[-limit:] if messages else []

    compact = []
    for m in tail:
        compact.append({
            "type": m.get("type"),
            "status": m.get("status"),
            "from": m.get("from"),
            "messageContent": str(m.get("messageContent", ""))[:3000],
            "createdDateTime": m.get("createdDateTime"),
        })
    return compact


def build_reviewer_input(
    task: dict,
    details: dict,
    messages: List[dict],
) -> str:
    parts = [
        REVIEWER_AGENT_PROMPT,
        "\nTASK:\n" + json.dumps(task, indent=2, ensure_ascii=False, default=str),
        "\nINTERVENTION DETAILS:\n" + json.dumps(details, indent=2, ensure_ascii=False, default=str),
        "\nRECENT MESSAGES:\n" + json.dumps(messages, indent=2, ensure_ascii=False, default=str),
    ]
    return "\n".join(parts)


def flatten_text(details: dict, messages: List[dict]) -> str:
    return (
        json.dumps(details, ensure_ascii=False).lower()
        + "\n"
        + json.dumps(messages, ensure_ascii=False).lower()
    )


# -----------------------------
# Optional tiny rule layer
# -----------------------------
def precheck_obvious_instruction_needed(
    details: dict,
    messages: List[dict],
) -> Optional[ReviewDecision]:

    text = flatten_text(details, messages)

    # Example: SOA explicitly says it needs help, but its draft seems unrelated
    # This layer is intentionally conservative.
    if "clarification was required" in text and "order confirmation" in text:
        return ReviewDecision(
            action="give_instructions",
            rationale="The task should continue, but the current step appears to move to order confirmation before required clarification is resolved.",
            instruction="Do not confirm the order yet. First respond to the customer's missing clarification and continue the workflow only after the required details are resolved.",
            code="premature_order_confirmation",
            message="Current step appears to be ahead of the required workflow stage."
        )

    return None


# -----------------------------
# Fallback
# -----------------------------
def fallback_rule_based() -> ReviewDecision:
    return ReviewDecision(
        action="approve",
        rationale="Fallback: defaulting to continue-first approval.",
        code="approve_default",
        message="No reliable correction signal detected."
    )


# -----------------------------
# Main classifier
# -----------------------------
async def classify_intervention_llm(
    kernel: Kernel,
    task: dict,
    details: dict,
    messages: List[dict],
) -> ReviewDecision:
    # tiny conservative rule layer first
    forced = precheck_obvious_instruction_needed(details, messages)
    if forced is not None:
        return forced

    prompt = build_reviewer_input(task, details, messages)

    raw_output = await call_model(kernel, prompt)
    cleaned = strip_code_fences(raw_output)

    try:
        decision = ReviewDecision.model_validate_json(cleaned)

        if decision.action not in {"approve", "give_instructions"}:
            return fallback_rule_based()

        if not decision.rationale.strip():
            return fallback_rule_based()

        if decision.action == "give_instructions":
            if not decision.instruction or not decision.instruction.strip():
                return fallback_rule_based()

        return decision

    except Exception:
        print("[reviewer] parse failed, using fallback")
        print(raw_output)
        return fallback_rule_based()


# -----------------------------
# Public API
# -----------------------------
async def review_one_attention_step(
    client: AgentTaskClient,
    task_id: int,
) -> dict:
    task = client.get_task(task_id)
    if not task.get("needsAttention"):
        return {
            "action": "approve",
            "rationale": "Task does not require attention.",
            "code": "no_attention_needed",
            "message": "Task does not currently require reviewer attention."
        }

    details = client.get_intervention_details(task_id)
    messages = collect_recent_messages(client, task_id, limit=6)

    kernel = build_kernel()
    decision = await classify_intervention_llm(kernel, task, details, messages)

    print("📝 Reviewer decision:")
    print(json.dumps(decision.model_dump(), indent=2, ensure_ascii=False))

    if decision.action == "approve":
        result = client.approve_intervention(task_id)
        return {
            **decision.model_dump(),
            "executionResult": result,
        }

    if decision.action == "give_instructions":
        result = client.respond_to_intervention(task_id, decision.instruction or "")
        return {
            **decision.model_dump(),
            "executionResult": result,
        }

    return decision.model_dump()