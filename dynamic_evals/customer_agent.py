# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import yaml
import uuid
import asyncio
import json
import re
import argparse
import random
import time
import requests
from typing import Optional

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict
from agent_task_client import AgentTaskClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPACE_DIR = os.path.join(SCRIPT_DIR, "space")

AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = ""
SERVICE_ID = "customer-agent"

# -----------------------------
# Structured output
# -----------------------------
class Turn(BaseModel):
    from_: str
    subject: str
    body: str

class GeneratedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    body: str

class RuntimeFollowUpPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str  
    intent: str
    trigger: str
    outcome_manifest_stage: str 
    max_additional_turns: int
    grounding_requirements: List[str] = []

class GeneratedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    initial_turn: GeneratedTurn

class GeneratedTestCase(BaseModel):
    name: str
    description: str
    sender_name: str
    sender_email: str
    sender_type: str
    persona_id: str
    persona_display_name: str
    tone_id: str
    tone_display_name: str
    initial_turn: Turn
    follow_up_plan: dict
    expected_data: dict
    attachments: list = []

# -----------------------------
# Prompt
# -----------------------------
CUSTOMER_AGENT_PROMPT = """
You are simulating a CUSTOMER in a business email conversation with a Sales Order Agent (SOA) in Microsoft Dynamics 365 Business Central.

You are given:
1. A SCENARIO
2. A SELECTED OUTCOME
3. A SELECTED PERSONA
4. A SELECTED TONE
5. AVAILABLE ITEMS
6. A SELECTED SENDER
7. SCENARIO HINTS
8. AN EXPECTED FOLLOW-UP PLAN
9. OUTCOME-SPECIFIC HINTS

Your job is to generate a single TEST CASE from the CUSTOMER side.

Rules:
- Use the SELECTED SENDER exactly as provided.
- The generated email must come from the same sender.
- Do NOT invent another sender.
- Do NOT change the sender email.
- The sender signature must match the selected sender name.

- Safety requirement:
  The generated customer email must be safe for Azure OpenAI content filtering.
  Do NOT use abusive, insulting, threatening, harassing, hateful, profane, or demeaning language.
  Do NOT attack the company, support team, employees, or the sales agent.
  Rude or dissatisfied tones are allowed only as professional bluntness, impatience, or directness.
  Rude does NOT mean abusive, insulting, profane, hostile, or personally demeaning.
  If the sender is suspicious, express that through an unusual sender domain,
  unusual request pattern, vague business context, mild urgency, or a request to bypass normal workflow.
  Suspicious does NOT mean rude, hostile, abusive, or unsafe.
  Guardrail and annotation scenarios should use policy-safe triggers such as prompt-injection-like instructions,
  unsupported workflow requests, unknown sender context, or requests to bypass validation.
  The customer may be dissatisfied, but must remain professional and non-abusive.

- The interaction style, clarification behavior, context usage, and directness should reflect the SELECTED PERSONA.
- The wording, politeness level, emotional coloring, and formality should reflect the SELECTED TONE.
- Use a business email style unless the persona/tone imply a different style.
- Customers may refer to products naturally by product name, optionally with quantity, and may sometimes include an item number.
- Do NOT force all product mentions into a strict canonical format.
- The customer email may be written as a natural paragraph, a short list of requested items, or a mix of both, depending on the persona and tone.
- Use whichever structure feels natural for the customer's writing style.
- Generate ONLY the first customer email turn as initial_turn.
- If the scenario is runtime_grounded, DO NOT generate any future follow-up email text now.
- Future grounded follow-ups will be generated later at execution time after the SOA reply is available.

- If OUTCOME-SPECIFIC HINTS say the selected outcome must manifest in the initial turn,
  then at least one concrete requested line item in the initial_turn MUST be under-specified
  in a way that prevents the SOA from safely creating the quote without asking a clarification question.

- The ambiguity must be attached to a specific requested line item, not only to general wording.
- The ambiguity must come from a missing or unclear model, variant, bundle option,
  exact item mapping, color split, size, unit of measure, or another detail required
  to safely determine the requested item.
- Do NOT resolve the ambiguity inside the same line.

- Do NOT use self-resolving wording such as:
  "please quote both",
  "pick the standard one",
  "pick whatever is fastest",
  "use the best-selling variant",
  "either is fine",
  "whichever matches",
  or similar phrasing that allows the SOA to proceed without clarification.

- Keep most of the request understandable, but ensure that one or two requested line items
  genuinely require clarification.

- If OUTCOME-SPECIFIC HINTS indicate an intervention-triggering request in the initial turn,
  then the initial_turn must contain at least one requested line item or request element
  that cannot be safely resolved within the catalog or standard quote workflow.

- For intervention_required, prefer unknown or non-catalog item references,
  unresolvable product names, or request patterns that cannot be handled by ordinary clarification.

- Do NOT make the intervention-triggering line look like a normal clarification case
  that could be resolved by simply asking for a missing color, variant, or bundle option.

- If the allowed intervention trigger types include unresolved_unknown_item,
  generate at least one item line that refers to a product name or item that does not map to the catalog.

- If the allowed intervention trigger types include no_catalog_match_for_all_requested_items,
  ensure that the core requested items cannot be matched to the catalog.

- Avoid self-resolving or delegating wording such as "pick the standard one",
  "quote the regular version", or "whichever fits best", because these are not strong intervention triggers.

- If OUTCOME-SPECIFIC HINTS say the selected outcome belongs to a later follow-up turn,
  then the initial_turn should establish the correct business context first and should NOT prematurely express the later follow-up action.

- Catalog grounding requirement:
  AVAILABLE ITEMS is the only allowed product catalog.
  For valid quote creation scenarios, every requested product line MUST refer to an item from AVAILABLE ITEMS.
  Do NOT invent product names, item numbers, models, brands, SKUs, or product categories that are not present in AVAILABLE ITEMS.
  Prefer copying the exact product display name from AVAILABLE ITEMS.
  Customers usually refer to products by product name, not by item number.
  Use item numbers only occasionally, for example in at most one requested line, unless the scenario explicitly asks for mixed item-number references.
  If the scenario requires ambiguity or clarification, the ambiguity must still be based on one or more AVAILABLE ITEMS, unless the outcome is explicitly intervention_required for unknown/non-catalog items.

- Do NOT generate final expected_data.
- Do not include markdown fences.

Return a single JSON object with this shape only:
{
  "name": "...",
  "description": "...",
  "initial_turn": {
    "subject": "...",
    "body": "..."
  }
}
"""

FOLLOWUP_AGENT_PROMPT = """
You are simulating the CUSTOMER in an ongoing business email conversation with a Sales Order Agent (SOA) in Microsoft Dynamics 365 Business Central..

You are given:
1. A SCENARIO
2. A SELECTED OUTCOME
3. A SELECTED PERSONA
4. A SELECTED TONE
5. A SELECTED SENDER
6. A FOLLOW-UP PLAN
7. THE CONVERSATION THREAD
8. THE LATEST SOA REPLY
9. THE ORIGINAL CUSTOMER REQUEST

Your job is to generate the next single customer email turn.

Rules:
- The reply MUST be grounded in the latest SOA reply, the conversation thread, and THE ORIGINAL CUSTOMER REQUEST.
- Treat THE ORIGINAL CUSTOMER REQUEST as the customer's own intent and memory.
- The reply MUST directly respond to the SOA's latest request, question, proposal, or clarification.
- When the SOA asks for clarification, identify which part of THE ORIGINAL CUSTOMER REQUEST caused the clarification, then answer that specific question.
- Do NOT ask the SOA to clarify again unless the SOA reply is genuinely unreadable or contains no actionable question.
- If the SOA presents options for an ambiguous item, choose one concrete option from the SOA reply.
- If the original request used a broad family name such as "T-Shirts", "Rug", "Pizza", or "Soda", and the SOA asks which option to use, pick one concrete available option from the SOA reply.
- The follow-up should be a customer clarification, not a new quote request.
- Do NOT ignore the latest SOA reply.
- Do NOT produce a generic greeting-only or signature-only message.
- Do NOT produce filler such as only "Hi", "Thanks", "Best regards", or similar low-information text.
- Preserve the thread context naturally.
- Do NOT invent items, quote details, order details, or clarifications that are not supported by the SOA reply unless clearly implied by the scenario.
- The interaction style should reflect the SELECTED PERSONA.
- The wording and politeness should reflect the SELECTED TONE.
- Use the SELECTED SENDER identity exactly as given.

- If FOLLOW-UP PLAN intent is "reply_to_soa_clarification":
  You MUST answer the SOA's clarification request directly.
  Use THE ORIGINAL CUSTOMER REQUEST to understand what the customer was trying to buy.
  If the SOA asks the customer to choose from listed options, select one concrete option from the SOA reply.
  Prefer the first available option unless the original customer request implies a different preference.
  Include the selected product name and item number if the SOA provided them.
  Do NOT ask the SOA to clarify again.
  Do NOT reply with generic text such as "Could you clarify that?".
  Do NOT introduce a brand-new unrelated product request.
- If FOLLOW-UP PLAN intent is "request_clear_update_to_existing_quote", request a clear and actionable update to the existing quote.
- If FOLLOW-UP PLAN intent is "request_ambiguous_update_to_existing_quote", request an update to the existing quote but leave one important update detail ambiguous or missing so that the SOA should ask for clarification.
- If FOLLOW-UP PLAN intent is "request_unresolvable_update_to_existing_quote", request an update to the existing quote that cannot be safely completed or resolved.
- If FOLLOW-UP PLAN intent is "request_clear_update_to_existing_order", request a clear update to the existing order.
- If FOLLOW-UP PLAN intent is "request_ambiguous_update_to_existing_order", request an update to the existing order but leave one important detail ambiguous or missing.
- If FOLLOW-UP PLAN intent is "request_unresolvable_update_to_existing_order", request an update to the existing order that cannot be safely completed or resolved.
- If FOLLOW-UP PLAN intent is "approve_existing_quote_or_order", clearly approve or confirm the existing quote or order in thread context.
- If FOLLOW-UP PLAN intent is "reject_existing_quote_or_order", clearly reject, decline, or defer the existing quote or order in thread context.
- If FOLLOW-UP PLAN intent is "select_from_previous_soa_options", select from the options that the SOA presented in the latest SOA reply.
- If FOLLOW-UP PLAN intent is "request_quote_from_previous_suggestions", request a quote for one or more items that were offered by the SOA in the latest SOA reply.
- If FOLLOW-UP PLAN intent is "ask_for_unsupported_product_attribute", ask about a product attribute that is not present in the latest SOA reply.
- If FOLLOW-UP PLAN intent is "approve_quote_for_order_conversion", clearly approve the previously created quote so that order conversion can be attempted.
- If FOLLOW-UP PLAN intent is "request_unsupported_record_update", request an update to a non-editable field or unsupported record change.
- If FOLLOW-UP PLAN intent is "request_quote_from_internal_context", request a quote based on the product context previously established in the conversation, not by introducing a brand-new unrelated request.

- Do not turn a follow-up into a brand-new unrelated request.
- Do not include markdown fences.

Return a single JSON object with this shape only:
{
  "subject": "...",
  "body": "..."
}
"""

EXPLICIT_SELF_RESOLVING_PHRASES = [
    "please quote both",
    "quote both",
    "pick the standard one",
    "pick whatever is fastest",
    "use the best-selling variant",
    "either is fine",
    "whichever matches",
    "whichever is standard",
    "whatever is fastest",
    "please quote for both",
    "quote for both options",
    "open to suggestions",
    "whatever is standard",
    "appropriate option",
    "best one you have",
    "whichever fits best",
    "you figure it out",
    "quote the right setup for us",
]

# -----------------------------
# Kernel / Model
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

    max_attempts = 6
    base_delay_seconds = 2.0

    last_error = None

    for attempt in range(1, max_attempts + 1):
        chat_history = ChatHistory()
        chat_history.add_system_message(prompt_text)

        settings = AzureChatPromptExecutionSettings(
            temperature=0.2,
            max_tokens=1200,
        )

        try:
            response = await chat_service.get_chat_message_content(
                chat_history,
                settings=settings,
            )
            return response.content.strip()

        except Exception as e:
            last_error = e
            msg = str(e)

            retryable = (
                "429" in msg
                or "RateLimit" in msg
                or "rate limit" in msg.lower()
                or "Backend error" in msg
                or "service failed to complete the prompt" in msg
                or "timeout" in msg.lower()
                or "temporarily unavailable" in msg.lower()
            )

            if not retryable:
                raise

            if attempt == max_attempts:
                break

            delay = base_delay_seconds * (2 ** (attempt - 1))
            delay = min(delay, 30.0)
            delay += random.uniform(0, 1.5)

            print(
                f"[model retry] transient model error on attempt "
                f"{attempt}/{max_attempts}: {type(e).__name__}: {msg[:300]}"
            )
            print(f"[model retry] sleeping {delay:.1f}s before retry...")
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"Model call failed after {max_attempts} attempts. Last error: {last_error}"
    )


# -----------------------------
# YAML / helpers
# -----------------------------
def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_personas(path: str):
    data = load_yaml(path)
    return data.get("personas", [])

def load_tones(path: str):
    data = load_yaml(path)
    return data.get("tones", [])

def prepare_items_for_prompt(items: list, seed_key: str, max_items: int = 80) -> list:

    items_copy = list(items)
    rng = random.Random(seed_key + ":prompt_items")
    rng.shuffle(items_copy)
    return items_copy[:max_items]

def format_items_for_prompt(items: list, max_items: int = 50) -> str:
    rows = []
    for i in items[:max_items]:
        number = (i.get("number") or "").strip()
        display_name = (i.get("displayName") or "").strip()
        unit_price = i.get("unitPrice", "")
        uom = (i.get("baseUnitOfMeasureCode") or "").strip()

        line = display_name

        if number:
            line += f" (Item No: {number})"

        if unit_price != "":
            if uom:
                line += f" - {unit_price} {uom}"
            else:
                line += f" - {unit_price}"

        rows.append(line)

    return "\n".join(rows)

def normalize_item_schema(i: dict) -> dict:
    """Normalize item dict fields across different BC/SOA endpoints."""

    def first_nonempty(*values):
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    # Try known field names first
    number = first_nonempty(
        i.get("number"),
        i.get("itemNo"),
        i.get("itemNumber"),
        i.get("no"),
        i.get("No"),
        i.get("ItemNo"),
        i.get("Item_No"),
        i.get("item_no"),
    )

    # Fallback: scan raw JSON for AItem-style number
    if not number:
        raw_text = json.dumps(i, ensure_ascii=False)
        m = re.search(r"\bAItem-\d{4}\b", raw_text, flags=re.IGNORECASE)
        if m:
            number = m.group(0)

    if number:
        number = number.upper()

    display_name = first_nonempty(
        i.get("displayName"),
        i.get("description"),
        i.get("itemDescription"),
        i.get("name"),
        i.get("DisplayName"),
        i.get("Description"),
    )

    base_uom = first_nonempty(
        i.get("baseUnitOfMeasureCode"),
        i.get("baseUnitOfMeasure"),
        i.get("salesUnitOfMeasure"),
        i.get("uomCode"),
        i.get("unitOfMeasureCode"),
        i.get("BaseUnitOfMeasure"),
        i.get("baseUom"),
    )

    unit_price = i.get("unitPrice")
    if unit_price in (None, ""):
        unit_price = i.get("price")
    if unit_price in (None, ""):
        unit_price = i.get("UnitPrice")
    if unit_price is None:
        unit_price = ""

    return {
        "number": number,
        "displayName": display_name,
        "unitPrice": unit_price,
        "baseUnitOfMeasureCode": base_uom,
        "_raw": i,
    }

def clean_paragraph_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()

def build_signature(sender_name: str) -> str:
    return normalize_message_newlines(f"Best regards,\n{sender_name}")

def enforce_sender_signature(body: str, sender_name: str) -> str:
    body = clean_paragraph_text(body or "")
    signature = build_signature(sender_name)

    if not body:
        return signature

    body = re.sub(
        r"(\n\s*(Best regards|Kind regards|Regards|Sincerely|Thanks|Thank you)[\s\S]*)$",
        "",
        body,
        flags=re.IGNORECASE,
    ).rstrip()

    return normalize_message_newlines(f"{body}\n\n{signature}")

def build_initial_email_message(turn: dict, attachments: list | None = None) -> str:
    subject = clean_paragraph_text((turn.get("subject") or "").strip())
    body = clean_paragraph_text((turn.get("body") or "").strip())

    if subject:
        return normalize_message_newlines(f"SUBJECT: {subject}\n\n{body}")
    return normalize_message_newlines(body)

def build_followup_message(turn: dict) -> str:
    body = clean_paragraph_text((turn.get("body") or "").strip())
    return normalize_message_newlines(body)    

def normalize_catalog_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r'["“”\'`]', " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def canonical_item_reference(
    description: Optional[str],
    item_no: Optional[str]
) -> str:

    description = (description or "").strip()
    item_no = (item_no or "").strip()

    if item_no and description:
        return f"{item_no} {description}"
    if item_no:
        return item_no
    return description

_ITEM_SUFFIX_PATTERNS = [
    re.compile(
        r"^(?P<desc>.+?)\s*\(\s*item\s+(?P<item>[A-Za-z0-9\-]+)\s*\)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<desc>.+?)\s*\(\s*(?P<item>[A-Za-z0-9\-]+)\s*\)$",
        re.IGNORECASE,
    ),
]

def normalize_item_reference(text: str) -> str:

    if not text:
        return text

    s = " ".join(text.strip().split())

    for pattern in _ITEM_SUFFIX_PATTERNS:
        m = pattern.match(s)
        if m:
            desc = m.group("desc").strip()
            item = m.group("item").strip()
            return canonical_item_reference(desc, item)

    return s


def normalize_inline_item_suffixes(body: str) -> str:

    if not body:
        return body

    body = re.sub(
        r'(?P<qty>\b\d+\b\s+)(?P<desc>[A-Za-z][^,\n]+?),\s*item\s+(?P<item>[A-Za-z0-9\-]+)',
        lambda m: f'{m.group("qty")}{canonical_item_reference(m.group("desc").strip(), m.group("item").strip())}',
        body,
        flags=re.IGNORECASE,
    )

    return body


def normalize_item_references_in_body(body: str) -> str:

    if not body:
        return body

    lines = body.splitlines()
    fixed_lines = []

    for line in lines:
        original_line = line
        stripped = line.strip()

        m = re.match(
            r'^(?P<prefix>\s*(?:(?:[-*])|\d+[.)])?\s*\d+\s*(?:x|units?\s+of|pcs?\s+of|pieces?\s+of|boxes?\s+of|packs?\s+of)\s+)'
            r'(?P<desc>.+?)\s*\(\s*(?:item\s+)?(?P<item>[A-Za-z0-9\-]+)\s*\)\s*$',
            stripped,
            re.IGNORECASE,
        )
        if m:
            prefix = m.group("prefix")
            desc = m.group("desc").strip()
            item = m.group("item").strip()
            fixed_lines.append(f"{prefix}{canonical_item_reference(desc, item)}")
            continue

        m2 = re.match(
            r'^(?P<prefix>\s*(?:[-*]|\d+[.)])\s*)'
            r'(?P<desc>.+?)\s*\(\s*(?:item\s+)?(?P<item>[A-Za-z0-9\-]+)\s*\)\s*$',
            stripped,
            re.IGNORECASE,
        )
        if m2:
            prefix = m2.group("prefix")
            desc = m2.group("desc").strip()
            item = m2.group("item").strip()
            fixed_lines.append(f"{prefix}{canonical_item_reference(desc, item)}")
            continue


        normalized_single = normalize_item_reference(stripped)
        if normalized_single != stripped:
            # preserve original indentation if any
            leading_ws = re.match(r'^(\s*)', original_line).group(1)
            fixed_lines.append(f"{leading_ws}{normalized_single}")
            continue

        fixed_lines.append(line)

    normalized = "\n".join(fixed_lines)
    normalized = normalize_inline_item_suffixes(normalized)
    return normalized


def has_noncanonical_item_suffix(body: str) -> bool:

    if not body:
        return False

    patterns = [
        re.compile(r'\(\s*item\s+[A-Za-z0-9\-]+\s*\)', re.IGNORECASE),
        re.compile(r',\s*item\s+[A-Za-z0-9\-]+', re.IGNORECASE),
    ]

    return any(p.search(body) for p in patterns)

def build_catalog_signatures(items: list[dict]) -> list[dict]:

    signatures = []

    for item in items:
        number = normalize_catalog_text(str(item.get("number", "")))
        display_name = normalize_catalog_text(str(item.get("displayName", "")))

        tokens = set(display_name.split())

        # keep only meaningful tokens
        stopwords = {
            "the", "and", "of", "for", "with", "inch", "in", "to",
            "unit", "units", "piece", "pieces", "pack", "packs"
        }
        tokens = {t for t in tokens if len(t) >= 3 and t not in stopwords}

        signatures.append({
            "number": number,
            "display_name": display_name,
            "tokens": tokens,
        })

    return signatures

def looks_like_low_information_followup(body: str, previous_soa_reply: str) -> bool:
    text = (body or "").strip().lower()

    if not text:
        return True

    # A short answer with a concrete item number is still useful.
    if re.search(r"\bAITEM-\d{4}\b", text, flags=re.IGNORECASE):
        return False

    # A short answer that clearly selects an option is also useful.
    selection_signals = [
        "please use",
        "please include",
        "use the first",
        "include the first",
        "go with",
        "select",
        "choose",
    ]
    if any(s in text for s in selection_signals):
        return False

    # Remove signature-ish endings for validation.
    text_wo_sig = re.sub(
        r"(best regards|kind regards|regards|sincerely|thanks|thank you)[\s\S]*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text_wo_sig:
        return True

    weak_patterns = [
        "hi there",
        "hello",
        "thanks",
        "thank you",
        "please proceed",
        "best regards",
    ]

    if len(text_wo_sig) < 20:
        return True

    if text_wo_sig in weak_patterns:
        return True

    if "?" in (previous_soa_reply or "") and len(text_wo_sig.split()) < 5:
        return True

    return False

def extract_option_from_soa_reply(previous_soa_reply: str) -> dict:

    text = previous_soa_reply or ""

    for line in text.splitlines():
        line_clean = " ".join(line.strip().split())
        if not line_clean:
            continue

        m = re.search(r"\b(AITEM-\d{4})\b", line_clean, flags=re.IGNORECASE)
        if not m:
            continue

        item_no = m.group(1).upper()

        after = line_clean[m.end():].strip(" -:\t")
        description = ""

        if after:
            stop_match = re.search(
                r"\b(Available|Unavailable|Not Available|£|\$|EUR|PCS|BOX|PACK|SET)\b",
                after,
                flags=re.IGNORECASE,
            )
            if stop_match:
                description = after[:stop_match.start()].strip(" -:\t")
            else:
                description = after.strip(" -:\t")

        return {
            "item_no": item_no,
            "description": description,
        }

    return {}


def build_clarification_answer_fallback(
    previous_soa_reply: str,
    selected_sender: dict,
    follow_up_plan: dict,
    original_customer_request: str = "",
) -> Turn:

    sender_name = selected_sender["name"]
    sender_email = selected_sender["email"]

    option = extract_option_from_soa_reply(previous_soa_reply)

    if option:
        item_no = option.get("item_no", "")
        description = option.get("description", "")

        if description:
            body = (
                f"Please use {description} ({item_no}) for the ambiguous item in my original request.\n\n"
                f"Best regards,\n{sender_name}"
            )
        else:
            body = (
                f"Please use item {item_no} for the ambiguous item in my original request.\n\n"
                f"Best regards,\n{sender_name}"
            )

        return Turn(
            from_=sender_email,
            subject="Re:",
            body=normalize_message_newlines(body),
        )

    body = (
        "Please use the first available option you listed for the ambiguous item in my original request.\n\n"
        f"Best regards,\n{sender_name}"
    )

    return Turn(
        from_=sender_email,
        subject="Re:",
        body=normalize_message_newlines(body),
    )


def is_bad_clarification_followup(body: str, follow_up_plan: dict) -> bool:

    intent = (follow_up_plan.get("intent") or "").lower()
    if intent != "reply_to_soa_clarification":
        return False

    text = (body or "").lower()

    bad_phrases = [
        "could you please clarify",
        "can you clarify",
        "please clarify that",
        "clarify that a bit more",
        "i need more information from you",
        "what do you mean",
        "could you provide more details",
        "please provide more details",
    ]

    return any(p in text for p in bad_phrases)

def strip_leading_quantity_and_symbols(line: str) -> str:
    line = (line or "").strip().lower()

    # remove list markers / enumerators:
    # "- ..."
    # "* ..."
    # "1. ..."
    # "2) ..."
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)

    # remove leading quantity expressions:
    # "10x ..."
    # "3 units of ..."
    # "5 pcs of ..."
    line = re.sub(
        r"^\s*\d+\s*(x|units?\s+of|pcs?\s+of|pieces?\s+of|boxes?\s+of|packs?\s+of)\b",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # fallback: remove just a bare leading integer if still present
    line = re.sub(r"^\s*\d+\s+", "", line, flags=re.IGNORECASE)

    return line.strip()


def line_matches_catalog_item(line: str, catalog_signatures: list[dict]) -> bool:

    raw = normalize_catalog_text(strip_leading_quantity_and_symbols(line))

    if not raw:
        return False

    # 1) exact item number mention
    for sig in catalog_signatures:
        if sig["number"] and sig["number"] in raw:
            return True

    line_tokens = set(raw.split())

    # 2) exact display name match or strong token overlap
    for sig in catalog_signatures:
        if sig["display_name"] and sig["display_name"] in raw:
            return True

        sig_tokens = sig["tokens"]
        if not sig_tokens:
            continue

        overlap = len(line_tokens & sig_tokens)

        # If line mentions 2+ meaningful tokens from a catalog item, treat as likely match
        if overlap >= 2:
            return True

        # If display name is short, allow 1 meaningful token only when an item number is also present
        if overlap >= 1 and any(ch.isdigit() for ch in raw) and sig["number"]:
            return True

    return False


def line_catalog_candidates(line: str, catalog_signatures: list[dict]) -> list[dict]:

    raw = normalize_catalog_text(strip_leading_quantity_and_symbols(line))
    if not raw:
        return []

    line_tokens = set(raw.split())
    candidates = []

    for sig in catalog_signatures:
        if not sig["display_name"] and not sig["number"]:
            continue

        matched = False

        # strong match: explicit item number
        if sig["number"] and sig["number"] in raw:
            matched = True

        # strong match: exact display name substring
        elif sig["display_name"] and sig["display_name"] in raw:
            matched = True

        else:
            sig_tokens = sig["tokens"]
            if sig_tokens:
                overlap = len(line_tokens & sig_tokens)

                # 2+ meaningful overlapping tokens => candidate
                if overlap >= 2:
                    matched = True

                # allow 1 token overlap only for short/generic lines
                elif overlap == 1 and len(line_tokens) <= 4:
                    matched = True

        if matched:
            candidates.append(sig)

    # de-dupe
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["number"], c["display_name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped


def line_catalog_candidate_count(line: str, items: list[dict]) -> int:
    signatures = build_catalog_signatures(items)
    return len(line_catalog_candidates(line, signatures))


def line_is_uniquely_resolved_by_catalog(line: str, items: list[dict]) -> bool:
    return line_catalog_candidate_count(line, items) == 1


def line_is_ambiguous_in_catalog(line: str, items: list[dict]) -> bool:
    return line_catalog_candidate_count(line, items) > 1


def line_has_no_catalog_match(line: str, items: list[dict]) -> bool:
    return line_catalog_candidate_count(line, items) == 0


def line_looks_like_unknown_or_non_catalog(line: str, catalog_signatures: list[dict]) -> bool:

    raw = normalize_catalog_text(strip_leading_quantity_and_symbols(line))
    if not raw:
        return False

    # If it matches catalog, it's not unknown
    if line_matches_catalog_item(raw, catalog_signatures):
        return False

    # Generic product-ish nouns that often indicate an item request
    productish_terms = [
        "desk", "system", "station", "setup", "solution", "bundle",
        "chair", "table", "storage", "lamp", "monitor", "pedestal",
        "computer", "panel", "whiteboard", "paint"
    ]

    if any(t in raw for t in productish_terms):
        return True

    # Brand/product style multi-word title case-ish requests often become non-catalog
    words = raw.split()
    if len(words) >= 2 and not has_item_number(raw):
        return True

    return False

def has_real_clarification_gap(
    scenario: dict,
    outcome: dict,
    initial_turn_text: str,
    items: list[dict],
) -> bool:
    if not is_clarification_outcome(outcome):
        return True

    stage = infer_outcome_manifest_stage(scenario, outcome)
    if stage != "initial_turn":
        return True

    text = (initial_turn_text or "").lower()
    lines = extract_request_lines(initial_turn_text, items)

    if any(sig in text for sig in EXPLICIT_SELF_RESOLVING_PHRASES):
        return False

    declared_gap_types = get_clarification_gap_types(scenario)

    mapping_based_gap_types = {
        "missing_variant",
        "partial_product_name",
        "missing_exact_item_mapping",
        "multiple_candidate_item_match",
        "base_description_without_required_attribute",
    }

    if declared_gap_types:
        for line in lines:
            for gap_type in declared_gap_types:
                if not matches_gap_type(line, gap_type):
                    continue

                if gap_type in mapping_based_gap_types:
                    candidate_count = line_catalog_candidate_count(line, items)

                    if candidate_count == 1:
                        continue

                    if candidate_count == 0:
                        continue

                    if candidate_count > 1:
                        return True
                else:
                    return True

        return False

    strong_signals = [
        "not sure",
        "please confirm",
        "please advise",
        "if available",
        "same as last time",
        "the standard one",
        "the larger one",
        "the usual model",
        "not sure which one",
        "not sure if",
        "i think",
        "assorted colors",
        "preferably",
        "need both",
        "which options are available",
        "what options are available",
        "same color as last time",
        "the one we discussed",
    ]
    if any(sig in text for sig in strong_signals):
        return True

    return False


def build_invalid_senders(count: int = 20) -> list:
    return [
        {
            "name": f"Unknown Sender {i}",
            "email": f"unknown.sender{i}@external.test",
            "sender_type": "unknown_but_plausible_sender",
        }
        for i in range(1, count + 1)
    ]


def build_suspicious_senders(count: int = 10) -> list:
    return [
        {
            "name": f"Suspicious Sender {i}",
            "email": f"suspicious.sender{i}@weird.test",
            "sender_type": "suspicious_sender",
        }
        for i in range(1, count + 1)
    ]

def scenario_requires_alternate_followup_sender(scenario: dict) -> bool:
    scenario_id = (scenario.get("scenario_id") or "").lower()
    capability = (scenario.get("capability") or "").lower()

    if scenario_id == "quote_approved_to_order_cross_contact_resolution":
        return True

    if "multiple_contact_emails" in capability or "alternate_contact_email" in capability:
        return True

    return False


def choose_alternate_sender(valid_customers: list, primary_sender: dict, seed_key: str) -> dict:
    rng = random.Random(seed_key + ":alternate_sender")

    candidates = [
        c for c in valid_customers
        if c.get("email", "").strip().lower() != primary_sender.get("email", "").strip().lower()
    ]

    if not candidates:
        return primary_sender

    return rng.choice(candidates)


def scenario_requires_attachments(scenario: dict) -> bool:
    scenario_id = (scenario.get("scenario_id") or "").lower()
    return scenario_id in {
        "quote_request_from_attachment",
        "inquiry_with_irrelevant_attachment",
    }

def build_customer_agent_prompt_for_scenario(scenario: dict) -> str:
    return CUSTOMER_AGENT_PROMPT

def build_case_attachments(scenario: dict, outcome: dict, items: list, case_id: str) -> list:
    scenario_id = (scenario.get("scenario_id") or "").lower()

    if scenario_id == "quote_request_from_attachment":
        return [
            {
                "file_name": f"{case_id}_items.pdf",
                "content_type": "application/pdf",
                "relevance": "relevant",
                "text_content": (
                    "10x 1908-S LONDON Swivel Chair\n"
                    "5x 1924-W CHAMONIX Base Storage Unit\n"
                    "2x 1928-S AMSTERDAM Lamp"
                )
            }
        ]

    if scenario_id == "inquiry_with_irrelevant_attachment":
        return [
            {
                "file_name": f"{case_id}_terms.pdf",
                "content_type": "application/pdf",
                "relevance": "irrelevant",
                "text_content": (
                    "Standard terms and conditions for office furniture supply.\n"
                    "This document does not contain any usable product request details."
                )
            }
        ]

    return []

def choose_sender_for_case(
    valid_customers: list,
    invalid_senders: list,
    suspicious_senders: list,
    sender_mode: str,
    seed_key: str,
) -> dict:
    rng = random.Random(seed_key)

    if sender_mode == "valid":
        return rng.choice(valid_customers)
    elif sender_mode == "invalid":
        return rng.choice(invalid_senders)
    elif sender_mode == "suspicious":
        return rng.choice(suspicious_senders)
    else:
        raise ValueError(f"Unknown sender_mode: {sender_mode}")


def strip_code_fences(text: str) -> str:
    return re.sub(r"^```json\s*|^```\s*|```$", "", text.strip(), flags=re.MULTILINE)


# -----------------------------
# Scenario hinting
# -----------------------------
def gc_list(gc: dict, key: str) -> list:
    value = gc.get(key, [])
    return value if isinstance(value, list) else []

def gc_str(gc: dict, key: str) -> str:
    value = gc.get(key, "")
    return value.strip().lower() if isinstance(value, str) else ""

def gc_list_str(gc: dict, key: str) -> list[str]:
    value = gc.get(key, [])
    return [str(x).strip().lower() for x in value] if isinstance(value, list) else []

def get_clarification_gap_scope(scenario: dict) -> str:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_str(gc, "clarification_gap_scope")

def get_clarification_gap_types(scenario: dict) -> list[str]:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_list_str(gc, "clarification_gap_types")

def get_intervention_manifest_scope(scenario: dict) -> str:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_str(gc, "intervention_manifest_scope")

def get_intervention_trigger_types(scenario: dict) -> list[str]:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_list_str(gc, "intervention_trigger_types")

def get_annotation_manifest_scope(scenario: dict) -> str:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_str(gc, "annotation_manifest_scope")

def get_annotation_trigger_types(scenario: dict) -> list[str]:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_list_str(gc, "annotation_trigger_types")

def get_follow_up_action_scope(scenario: dict) -> str:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_str(gc, "follow_up_action_scope")

def get_follow_up_action_intent(scenario: dict) -> str:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_str(gc, "follow_up_action_intent")

def get_unsupported_information_scope(scenario: dict) -> str:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_str(gc, "unsupported_information_scope")

def get_unsupported_information_types(scenario: dict) -> list[str]:
    gc = scenario.get("generation_constraints", {}) or {}
    return gc_list_str(gc, "unsupported_information_types")

def looks_like_request_segment(segment: str, items: list[dict]) -> bool:
    segment = (segment or "").strip()
    if not segment:
        return False

    # strong signals
    if extract_leading_quantity(segment) is not None:
        return True

    if has_item_number(segment):
        return True

    # if it plausibly maps to catalog, keep it
    if line_catalog_candidate_count(segment, items) > 0:
        return True

    return False

def split_request_sentence_into_item_chunks(sentence: str, items: list[dict]) -> list[str]:

    if not sentence:
        return []

    s = sentence.strip()

    # First split by semicolon / newline-like punctuation
    coarse_parts = re.split(r"[;；]+", s)
    chunks = []

    for part in coarse_parts:
        part = part.strip()
        if not part:
            continue

        # Further split by comma / and only if the piece still looks request-like
        subparts = re.split(r"\s+(?:and|,)\s+", part, flags=re.IGNORECASE)

        if len(subparts) == 1:
            if looks_like_request_segment(part, items):
                chunks.append(part)
            continue

        kept = []
        for sp in subparts:
            sp = sp.strip()
            if looks_like_request_segment(sp, items):
                kept.append(sp)

        if kept:
            chunks.extend(kept)
        elif looks_like_request_segment(part, items):
            chunks.append(part)

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for c in chunks:
        key = c.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped

def extract_request_lines(initial_turn_text: str, items: list[dict]) -> list[str]:

    if not initial_turn_text:
        return []

    results = []
    seen = set()

    raw_lines = [line.strip() for line in initial_turn_text.splitlines() if line.strip()]

    # --------------------------------------------------
    # Pass 1: explicit bullet / numbered list lines
    # --------------------------------------------------
    for raw_line in raw_lines:
        stripped = raw_line.strip()

        if stripped.startswith("-") or stripped.startswith("*") or re.match(r"^\d+[.)\s-]", stripped):
            normalized = stripped.lower()
            if normalized not in seen:
                seen.add(normalized)
                results.append(normalized)

    # --------------------------------------------------
    # Pass 2: paragraph / mixed content
    # --------------------------------------------------
    paragraph_text = " ".join(raw_lines)
    paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()

    if paragraph_text:
        # sentence-ish splitting
        sentence_parts = re.split(r"(?<=[\.\?!])\s+", paragraph_text)

        for sentence in sentence_parts:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If the whole sentence looks like a request segment, keep it or split it
            if looks_like_request_segment(sentence, items):
                chunks = split_request_sentence_into_item_chunks(sentence, items)
                if chunks:
                    for chunk in chunks:
                        normalized = chunk.strip().lower()
                        if normalized and normalized not in seen:
                            seen.add(normalized)
                            results.append(normalized)
                else:
                    normalized = sentence.lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        results.append(normalized)

    return results


def has_quantity(line: str) -> bool:
    return extract_leading_quantity(line) is not None



def has_item_number(line: str) -> bool:
    return bool(
        re.search(
            r"\b(?:AItem-\d{4}|\d{4,5}(?:-[A-Za-z])?)\b",
            line or "",
            re.IGNORECASE,
        )
    )


def has_bundle_option(line: str) -> bool:
    return "1-6" in line or "1-8" in line


def has_context_reference(line: str) -> bool:
    phrases = [
        "same as last time",
        "the one we discussed",
        "previous order",
        "usual model",
        "same color as last time",
        "same as we discussed",
        "same as before",
    ]
    return any(p in line for p in phrases)


def has_color_split_ambiguity(line: str) -> bool:
    phrases = [
        "both black and yellow",
        "both yellow and black",
        "black and yellow",
        "red and blue",
        "blue and green",
        "assorted colors",
        "mix of colors",
        "preferably red and blue",
        "need both",
    ]
    return any(p in line for p in phrases)

def looks_like_generic_product_family(line: str) -> bool:
    families = [
        "chair",
        "guest chair",
        "swivel chair",
        "storage unit",
        "conference bundle",
        "whiteboard",
        "paint",
        "monitor",
        "lamp",
        "pedestal",
    ]
    return any(f in line for f in families)

def has_specific_model_name(line: str) -> bool:
    models = [
        "paris",
        "london",
        "berlin",
        "rome",
        "tokyo",
        "sydney",
        "athens",
        "amsterdam",
        "chamonix",
        "innsbruck",
        "grenoble",
        "sapporo",
        "antwerp",
        "m780",
        "m009",
    ]
    return any(m in line for m in models)

def has_explicit_variant_or_attribute(line: str) -> bool:
    hints = [
        "blue",
        "green",
        "yellow",
        "black",
        "red",
        "with drawers",
        "with glass doors",
        "glass door",
        "drawers",
        "standard version",
        "t variant",
    ]
    return any(h in line for h in hints)


def matches_gap_type(line: str, gap_type: str) -> bool:
    gap_type = gap_type.lower().strip()
    line = (line or "").lower()

    # -----------------------------
    # Initial-turn quote clarification gaps
    # -----------------------------
    if gap_type == "missing_quantity":
        return looks_like_generic_product_family(line) and not has_quantity(line)

    if gap_type == "missing_uom":
        qty = extract_leading_quantity(line)
        has_uom = bool(re.search(r"\b(units?|pcs?|pieces?|boxes?|packs?)\b", line, re.IGNORECASE))
        explicit_x = bool(re.search(r"^\s*(?:[-*]\s*)?\d+\s*x\b", line, re.IGNORECASE))
        return qty is not None and not has_uom and not explicit_x

    if gap_type == "partial_product_name":
        return looks_like_generic_product_family(line) and not has_specific_model_name(line) and not has_item_number(line)

    if gap_type == "missing_variant":
        return (
            looks_like_generic_product_family(line)
            and not has_item_number(line)
            and not has_specific_model_name(line)
            and not has_explicit_variant_or_attribute(line)
        )

    if gap_type == "missing_attribute_value":
        return (
            looks_like_generic_product_family(line)
            and (
                "with" in line
                or "attribute" in line
                or "material" in line
                or "color" in line
            )
            and not has_explicit_variant_or_attribute(line)
        )

    if gap_type == "missing_bundle_option":
        return "conference bundle" in line and not has_bundle_option(line)

    if gap_type == "missing_exact_item_mapping":
        return looks_like_generic_product_family(line) and not has_item_number(line) and not has_specific_model_name(line)

    if gap_type == "ambiguous_color_split":
        return has_color_split_ambiguity(line)

    if gap_type == "context_dependent_reference":
        return has_context_reference(line)

    # -----------------------------
    # Attribute clarification
    # -----------------------------
    if gap_type == "unavailable_attribute_value":
        unavailable_words = ["purple", "orange", "magenta", "oak finish", "velvet"]
        return any(w in line for w in unavailable_words)

    if gap_type == "ambiguous_attribute_combination":
        return (
            "black and yellow" in line
            or "red and blue" in line
            or "mix of colors" in line
        )

    if gap_type == "partial_attribute_description":
        return (
            looks_like_generic_product_family(line)
            and ("material" in line or "finish" in line or "color" in line)
            and not has_explicit_variant_or_attribute(line)
        )

    if gap_type == "base_description_without_required_attribute":
        return looks_like_generic_product_family(line) and not has_explicit_variant_or_attribute(line)

    # -----------------------------
    # Attachment clarification
    # -----------------------------
    if gap_type == "insufficient_attachment_item_details":
        return "attached" in line or "see attachment" in line

    if gap_type == "unreadable_or_partial_item_lines":
        return "screenshot" in line or "image" in line

    if gap_type == "missing_quantity_in_attachment":
        return "attached" in line and "quantity" not in line

    if gap_type == "missing_item_identity_in_attachment":
        return "attached" in line and "item" not in line and "product" not in line

    if gap_type == "mixed_relevant_and_insufficient_attachment_content":
        return "attached" in line and "also attached" in line

    # -----------------------------
    # Shipping clarification
    # -----------------------------
    if gap_type == "incomplete_shipping_address":
        return "ship to" in line and not any(x in line for x in ["street", "road", "avenue", "city", "zip", "postal"])

    if gap_type == "ambiguous_shipping_location_reference":
        return "our usual warehouse" in line or "the other office" in line or "alternate site" in line

    if gap_type == "missing_address_component":
        return "ship to" in line and ("," not in line or len(re.findall(r"\d", line)) == 0)

    if gap_type == "unclear_alternate_ship_to_target":
        return "alternate address" in line or "other site" in line

    # -----------------------------
    # Availability / CTP clarification
    # -----------------------------
    if gap_type == "ambiguous_item_selection":
        return "any of those" in line or "whichever is available" in line

    if gap_type == "ambiguous_shipment_option":
        return "earliest option" in line or "best shipment option" in line

    if gap_type == "ambiguous_availability_preference":
        return "as soon as possible" in line and "date" not in line

    if gap_type == "multiple_candidate_item_match":
        return looks_like_generic_product_family(line) and not has_item_number(line) and not has_specific_model_name(line)

    if gap_type == "missing_requested_date_for_option_selection":
        return (
            ("delivery" in line or "shipment" in line)
            and not bool(re.search(r"\b(today|tomorrow|next week|next month|july|august|\d{1,2}/\d{1,2})\b", line))
        )

    # -----------------------------
    # Update / follow-up clarification gaps
    # -----------------------------
    if gap_type == "ambiguous_line_reference":
        phrases = [
            "that item",
            "the first line",
            "the second one",
            "the chair",
            "the storage one",
            "that one",
            "the previous one",
        ]
        return any(p in line for p in phrases)

    if gap_type == "missing_update_target":
        phrases = [
            "please update it",
            "change it",
            "modify that",
            "adjust the quote",
        ]
        return any(p in line for p in phrases)

    if gap_type == "missing_new_value":
        phrases = [
            "please change it",
            "update the quantity",
            "switch the item",
            "replace it",
        ]
        return any(p in line for p in phrases) and not bool(re.search(r"\bto\b", line))

    if gap_type == "missing_replacement_item":
        phrases = [
            "replace it",
            "switch the item",
            "use the other one",
            "change to another model",
        ]
        return any(p in line for p in phrases) and not has_specific_model_name(line) and not has_item_number(line)

    if gap_type == "missing_new_quantity":
        return "quantity" in line and not bool(re.search(r"\bto\s+\d+\b", line))

    if gap_type == "ambiguous_add_remove_instruction":
        phrases = [
            "add another",
            "remove one",
            "add one more",
            "take one out",
        ]
        return any(p in line for p in phrases)

    if gap_type == "missing_new_uom":
        return "change the unit" in line and not any(u in line for u in ["pcs", "pieces", "boxes", "packs", "units"])

    if gap_type == "missing_edit_target":
        return "please update" in line and not any(p in line for p in ["chair", "table", "line", "item", "bundle"])

    if gap_type == "ambiguous_edit_instruction":
        return "adjust that one" in line or "change the previous item" in line

    if gap_type == "missing_revised_shipment_preference":
        return "please revise delivery" in line and not bool(re.search(r"\b(today|tomorrow|next week|july|august|\d{1,2}/\d{1,2})\b", line))

    if gap_type == "ambiguous_quantity_change":
        return "increase it" in line or "reduce it" in line

    if gap_type == "ambiguous_fulfillment_preference":
        return "whatever shipment works best" in line or "earliest possible is fine" in line

    if gap_type == "missing_confirmation_of_new_shipment_date":
        return "the new shipment date is okay" in line and not bool(re.search(r"\b(today|tomorrow|next week|july|august|\d{1,2}/\d{1,2})\b", line))

    return False

def has_attachment_consistency(scenario: dict, generated_case: GeneratedCase) -> bool:
    scenario_id = (scenario.get("scenario_id") or "").lower()
    body = (generated_case.initial_turn.body or "").lower()
    attachments = generated_case.attachments or []

    # Non-attachment scenarios: require no attachments
    if scenario_id not in {"quote_request_from_attachment", "inquiry_with_irrelevant_attachment"}:
        return len(attachments) == 0

    # Attachment scenarios should have at least one attachment
    if len(attachments) == 0:
        return False

    # Body mentions two attachments -> require two
    if "two documents" in body or "two attachments" in body:
        if len(attachments) != 2:
            return False

    # Body mentions PDF
    if "pdf" in body:
        if not any(a.content_type == "application/pdf" for a in attachments):
            return False

    # Body mentions image
    if "image" in body or "png" in body or "jpg" in body or "jpeg" in body:
        if not any(a.content_type.startswith("image/") for a in attachments):
            return False

    return True

def has_real_intervention_trigger(
    scenario: dict,
    outcome: dict,
    initial_turn_text: str,
    items: list[dict],
) -> bool:
    if not is_intervention_outcome(outcome):
        return True

    stage = infer_outcome_manifest_stage(scenario, outcome)
    if stage != "initial_turn":
        return True

    text = (initial_turn_text or "").lower()
    lines = extract_request_lines(initial_turn_text, items)
    trigger_types = get_intervention_trigger_types(scenario)

    if not trigger_types:
        return True

    if any(sig in text for sig in EXPLICIT_SELF_RESOLVING_PHRASES):
        return False

    catalog_signatures = build_catalog_signatures(items)

    clarification_like_signals = [
        "guest chair",
        "conference bundle",
        "storage unit with glass doors",
        "same as last time",
        "same as before",
        "not sure which one",
        "please confirm",
        "please advise",
    ]

    if "unresolved_unknown_item" in trigger_types:
        for line in lines:
            if line_looks_like_unknown_or_non_catalog(line, catalog_signatures):
                return True

    if "no_catalog_match_for_all_requested_items" in trigger_types:
        product_lines = [line for line in lines if strip_leading_quantity_and_symbols(line)]
        if product_lines:
            unmatched = 0
            for line in product_lines:
                if line_looks_like_unknown_or_non_catalog(line, catalog_signatures):
                    unmatched += 1
            if unmatched >= max(1, len(product_lines) // 2):
                return True

    unsafe_signals = [
        "you decide which one",
        "use whatever system fits best",
        "quote the right setup for us",
        "just figure out the correct product",
        "whatever is standard",
        "best one you have",
    ]
    if "unsafe_or_unprocessable_initial_quote_request" in trigger_types:
        if any(sig in text for sig in unsafe_signals):
            return True

    if lines and all(any(sig in line.lower() for sig in clarification_like_signals) for line in lines):
        return False

    return False


def extract_leading_quantity(line: str) -> Optional[int]:
    """
    Extract a likely requested quantity from the beginning of a request line.
    Handles numbered list prefixes such as:
      - "1. 4 units of ATHENS Mobile Pedestal" -> 4
      - "- 10x LONDON Swivel Chair" -> 10
      - "1908-S LONDON Swivel Chair" -> None
    """
    line = (line or "").strip()

    # remove list prefix first:
    # "1. ..."
    # "2) ..."
    # "- ..."
    # "* ..."
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)

    m = re.search(
        r"^\s*(?P<qty>\d+)\s*(x|units?\b|pcs?\b|pieces?\b|boxes?\b|packs?\b)(?:\s+of)?\b",
        line,
        re.IGNORECASE,
    )
    if m:
        try:
            return int(m.group("qty"))
        except Exception:
            return None

    # fallback: lines like "3 Guest Chair"
    m2 = re.search(
        r"^\s*(?P<qty>\d+)\s+",
        line,
        re.IGNORECASE,
    )
    if m2:
        try:
            return int(m2.group("qty"))
        except Exception:
            return None

    return None


def build_scenario_hints(scenario: dict, outcome: dict) -> str:
    gc = scenario.get("generation_constraints", {})
    hints = []

    # high-level intent / workflow
    hints.append(f"Selected scenario_id: {scenario.get('scenario_id')}")
    hints.append(f"Selected outcome_id: {outcome.get('outcome_id')}")

    # customer turns constraint
    customer_turns = scenario.get("customer_turns", {})
    hints.append(
        f"Customer turns must be between {customer_turns.get('min', 1)} and {customer_turns.get('max', 1)}."
    )

    # request patterns
    req_patterns = gc_list(gc, "request_patterns")
    if req_patterns:
        hints.append("Request patterns to reflect:")
        for p in req_patterns:
            hints.append(f"- {p}")

    # item resolution patterns
    item_patterns = gc_list(gc, "item_resolution_patterns")
    if item_patterns:
        hints.append("Item resolution patterns:")
        for p in item_patterns:
            hints.append(f"- {p}")

    # variant patterns
    variant_patterns = gc_list(gc, "variant_resolution_patterns")
    if variant_patterns:
        hints.append("Variant resolution patterns:")
        for p in variant_patterns:
            hints.append(f"- {p}")

    # uom patterns
    uom_patterns = gc_list(gc, "uom_resolution_patterns")
    if uom_patterns:
        hints.append("UOM resolution patterns:")
        for p in uom_patterns:
            hints.append(f"- {p}")

    # delivery
    delivery_patterns = gc_list(gc, "delivery_requirement_patterns")
    if delivery_patterns:
        hints.append("Delivery requirement patterns:")
        for p in delivery_patterns:
            hints.append(f"- {p}")

    # social tone / pricing hints
    social_patterns = gc_list(gc, "social_noise_patterns")
    if social_patterns:
        hints.append("Social/tone patterns:")
        for p in social_patterns:
            hints.append(f"- {p}")

    pricing_patterns = gc_list(gc, "pricing_request_patterns")
    if pricing_patterns:
        hints.append("Pricing/budget patterns:")
        for p in pricing_patterns:
            hints.append(f"- {p}")

    # follow-up grounding
    follow_up_grounding = gc_list(gc, "follow_up_grounding")
    if follow_up_grounding:
        hints.append("Follow-up grounding rules:")
        for p in follow_up_grounding:
            hints.append(f"- {p}")

    # review grounding
    review_grounding = gc.get("review_grounding")
    if review_grounding:
        hints.append("Review grounding rules:")
        hints.append(yaml.dump(review_grounding, allow_unicode=True))

    # intervention grounding
    intervention_grounding = gc.get("intervention_grounding")
    if intervention_grounding:
        hints.append("Intervention grounding rules:")
        hints.append(yaml.dump(intervention_grounding, allow_unicode=True))

    # clarification behavior
    clarification_behavior = gc_list(gc, "clarification_behavior")
    if clarification_behavior:
        hints.append("Clarification behavior:")
        for p in clarification_behavior:
            hints.append(f"- {p}")

    # response behavior
    response_behavior = gc_list(gc, "response_behavior")
    if response_behavior:
        hints.append("Response behavior:")
        for p in response_behavior:
            hints.append(f"- {p}")

    clarification_gap_scope = gc_str(gc, "clarification_gap_scope")
    if clarification_gap_scope:
        hints.append(f"Clarification gap scope: {clarification_gap_scope}")

    clarification_gap_types = gc_list_str(gc, "clarification_gap_types")
    if clarification_gap_types:
        hints.append("Clarification gap types:")
        for p in clarification_gap_types:
            hints.append(f"- {p}")

    intervention_manifest_scope = gc_str(gc, "intervention_manifest_scope")
    if intervention_manifest_scope:
        hints.append(f"Intervention manifest scope: {intervention_manifest_scope}")

    intervention_trigger_types = gc_list_str(gc, "intervention_trigger_types")
    if intervention_trigger_types:
        hints.append("Intervention trigger types:")
        for p in intervention_trigger_types:
            hints.append(f"- {p}")

    annotation_manifest_scope = gc_str(gc, "annotation_manifest_scope")
    if annotation_manifest_scope:
        hints.append(f"Annotation manifest scope: {annotation_manifest_scope}")

    annotation_trigger_types = gc_list_str(gc, "annotation_trigger_types")
    if annotation_trigger_types:
        hints.append("Annotation trigger types:")
        for p in annotation_trigger_types:
            hints.append(f"- {p}")

    follow_up_action_scope = gc_str(gc, "follow_up_action_scope")
    if follow_up_action_scope:
        hints.append(f"Follow-up action scope: {follow_up_action_scope}")

    follow_up_action_intent = gc_str(gc, "follow_up_action_intent")
    if follow_up_action_intent:
        hints.append(f"Follow-up action intent: {follow_up_action_intent}")

    unsupported_information_scope = gc_str(gc, "unsupported_information_scope")
    if unsupported_information_scope:
        hints.append(f"Unsupported information scope: {unsupported_information_scope}")

    unsupported_information_types = gc_list_str(gc, "unsupported_information_types")
    if unsupported_information_types:
        hints.append("Unsupported information types:")
        for p in unsupported_information_types:
            hints.append(f"- {p}")

    return "\n".join(hints)

def build_outcome_specific_hints(scenario: dict, outcome: dict) -> str:
    stage = infer_outcome_manifest_stage(scenario, outcome)
    intent = infer_follow_up_intent(scenario, outcome)

    hints = []
    hints.append(f"Outcome manifest stage: {stage}")
    hints.append(f"Follow-up intent: {intent}")

    if is_clarification_outcome(outcome):
        gap_types = get_clarification_gap_types(scenario)
        if stage == "initial_turn":
            hints.append("The initial customer email must contain a realistic ambiguity or missing detail.")
            hints.append("The ambiguity must be important enough that the SOA should ask for clarification.")
            if gap_types:
                hints.append("Allowed clarification gap types:")
                for g in gap_types:
                    hints.append(f"- {g}")
            hints.append("Do not make every line ambiguous. Keep the overall business intent clear.")
        else:
            hints.append("The initial customer email should mainly establish the context.")
            hints.append("The ambiguity that causes clarification should appear later in the runtime-generated follow-up.")
            if gap_types:
                hints.append("Allowed clarification gap types for the later follow-up:")
                for g in gap_types:
                    hints.append(f"- {g}")

    if is_intervention_outcome(outcome):
        trigger_types = get_intervention_trigger_types(scenario)
        if stage == "initial_turn":
            hints.append("The initial customer email should contain a request that triggers intervention.")
            hints.append("Prefer unknown, non-catalog, or unresolvable item references over ordinary clarification gaps.")
            hints.append("Do not make the issue look like a simple missing color, missing variant, or missing bundle option.")
            hints.append("At least one requested line should fail to map to the catalog, rather than merely lacking a small detail.")
        else:
            hints.append("The initial customer email should establish the context first.")
            hints.append("The later runtime-generated follow-up should trigger the intervention.")

        if trigger_types:
            hints.append("Allowed intervention trigger types:")
            for g in trigger_types:
                hints.append(f"- {g}")

    if is_annotation_outcome(outcome):
        trigger_types = get_annotation_trigger_types(scenario)
        if stage == "initial_turn":
            hints.append("The initial customer email should already contain content that should be annotated and stopped.")
        else:
            hints.append("The later follow-up should contain content that should be annotated and stopped.")
        if trigger_types:
            hints.append("Annotation trigger types:")
            for g in trigger_types:
                hints.append(f"- {g}")

    unsupported_scope = get_unsupported_information_scope(scenario)
    unsupported_types = get_unsupported_information_types(scenario)
    if unsupported_scope or unsupported_types:
        if unsupported_scope:
            hints.append(f"Unsupported information scope: {unsupported_scope}")
        if unsupported_types:
            hints.append("Unsupported information types:")
            for t in unsupported_types:
                hints.append(f"- {t}")
        hints.append("The customer may ask for unsupported information, but the request should remain valid for the scenario when applicable.")
        hints.append("Do not fabricate unsupported information and do not promise future delivery of missing unsupported details.")

    return "\n".join(hints)

def build_persona_prompt(persona: dict) -> str:
    return yaml.dump(
        {
            "persona_id": persona.get("persona_id"),
            "display_name": persona.get("display_name"),
            "description": persona.get("description"),
            "communication_style": persona.get("communication_style", {}),
            "interaction_style": persona.get("interaction_style", {}),
        },
        allow_unicode=True,
        sort_keys=False,
    )

def build_tone_prompt(tone: dict) -> str:
    return yaml.dump(
        {
            "tone_id": tone.get("tone_id"),
            "display_name": tone.get("display_name"),
            "description": tone.get("description"),
        },
        allow_unicode=True,
        sort_keys=False,
    )

def build_customer_agent_view(scenario: dict, outcome: dict):

    scenario_allowed_keys = [
        "scenario_id",
        "intent",
        "domain",
        "capability",
        "interaction_mode",
        "conversation_mode",
        "customer_turns",
        "workflow_template",
        "prerequisites",
        "goal",
        "generation_constraints",
        "sender_constraints",
    ]

    safe_scenario = {
        k: scenario[k]
        for k in scenario_allowed_keys
        if k in scenario
    }

    safe_outcome = {
        "outcome_id": outcome.get("outcome_id"),
        "description": outcome.get("description", ""),
    }

    return safe_scenario, safe_outcome

def flatten_strings(obj) -> list[str]:
    values = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            values.append(str(k))
            values.extend(flatten_strings(v))
    elif isinstance(obj, list):
        for x in obj:
            values.extend(flatten_strings(x))
    elif obj is not None:
        values.append(str(obj))

    return values

def is_clarification_outcome(outcome: dict) -> bool:
    outcome_id = (outcome.get("outcome_id") or "").lower()
    expected = outcome.get("expected", {}) or {}
    return (
        outcome_id == "clarification_required"
        or bool(expected.get("clarification_email"))
    )

def is_intervention_outcome(outcome: dict) -> bool:
    outcome_id = (outcome.get("outcome_id") or "").lower()
    expected = outcome.get("expected", {}) or {}
    return (
        "intervention" in outcome_id
        or bool(expected.get("userIntervention"))
    )

def is_annotation_outcome(outcome: dict) -> bool:
    outcome_id = (outcome.get("outcome_id") or "").lower()
    expected = outcome.get("expected", {}) or {}
    return (
        "annotation" in outcome_id
        or bool(expected.get("annotations"))
    )

def infer_outcome_manifest_stage(scenario: dict, outcome: dict) -> str:
    """
    Decide whether the selected outcome should be expressed in:
    - initial_turn
    - follow_up_turn
    - initial_turn_or_follow_up_turn
    """
    # 1) explicit YAML declaration wins
    if is_clarification_outcome(outcome):
        scope = get_clarification_gap_scope(scenario)
        if scope:
            return scope

    if is_intervention_outcome(outcome):
        scope = get_intervention_manifest_scope(scenario)
        if scope:
            return scope

    if is_annotation_outcome(outcome):
        scope = get_annotation_manifest_scope(scenario)
        if scope:
            return scope

    # 2) explicit follow-up action can also imply follow-up
    follow_up_scope = get_follow_up_action_scope(scenario)
    if follow_up_scope:
        return follow_up_scope

    # 3) fallback to your old heuristic
    outcome_id = (outcome.get("outcome_id") or "").lower()

    prereq = scenario.get("prerequisites", {}) or {}
    requires_existing_quote = bool(prereq.get("requires_existing_quote"))
    requires_existing_order = bool(prereq.get("requires_existing_order"))

    gc = scenario.get("generation_constraints", {}) or {}
    tokens = [t.lower() for t in flatten_strings(gc)]
    token_blob = " ".join(tokens)

    goal = (scenario.get("goal") or "").lower()
    capability = (scenario.get("capability") or "").lower()

    followup_signals = [
        "existing quote",
        "existing order",
        "quote created by soa",
        "sales order created by soa",
        "post_order_update",
        "modify_existing_quote",
        "update_existing_quote",
        "update_quote",
        "apply_requested_quote_updates",
        "post_quote_change_request",
        "post_order_edit_request",
        "approve_the_existing_quote",
        "after the quote has been created",
        "after the quote is created",
        "later follow-up",
        "follow-up email",
        "quote review",
        "review step",
    ]

    if requires_existing_quote or requires_existing_order:
        return "follow_up_turn"

    if any(sig in goal for sig in followup_signals):
        return "follow_up_turn"

    if any(sig in capability for sig in followup_signals):
        return "follow_up_turn"

    if any(sig in token_blob for sig in followup_signals):
        return "follow_up_turn"

    return "initial_turn"

def infer_follow_up_intent(scenario: dict, outcome: dict) -> str:
    # 1) explicit YAML declaration wins
    explicit_intent = get_follow_up_action_intent(scenario)
    if explicit_intent:
        return explicit_intent

    # 2) intervention / annotation outcomes in initial_turn usually do not require customer follow-up
    if is_intervention_outcome(outcome):
        stage = infer_outcome_manifest_stage(scenario, outcome)
        if stage == "initial_turn":
            return "none"

    if is_annotation_outcome(outcome):
        stage = infer_outcome_manifest_stage(scenario, outcome)
        if stage == "initial_turn":
            return "none"

    gc = scenario.get("generation_constraints", {}) or {}
    token_blob = " ".join([t.lower() for t in flatten_strings(gc)])
    capability = (scenario.get("capability") or "").lower()
    goal = (scenario.get("goal") or "").lower()
    intent = (scenario.get("intent") or "").lower()

    combined = " ".join([capability, goal, intent, token_blob])

    if "soa clarification" in combined or "previous_soa_clarification" in combined:
        return "reply_to_soa_clarification"

    if "update" in combined and "quote" in combined:
        if is_clarification_outcome(outcome):
            return "request_ambiguous_update_to_existing_quote"
        if is_intervention_outcome(outcome):
            return "request_unresolvable_update_to_existing_quote"
        return "request_clear_update_to_existing_quote"

    if "post_order_update" in combined or ("order" in combined and "update" in combined):
        if is_clarification_outcome(outcome):
            return "request_ambiguous_update_to_existing_order"
        if is_intervention_outcome(outcome):
            return "request_unresolvable_update_to_existing_order"
        return "request_clear_update_to_existing_order"

    if "approve" in combined or "approval" in combined or "confirm" in combined:
        return "approve_existing_quote_or_order"

    if "reject" in combined or "decline" in combined or "not approved" in combined:
        return "reject_existing_quote_or_order"

    if "select from previously offered options" in combined or "select_from_previously_offered_options" in combined:
        return "select_from_previous_soa_options"

    return "runtime_grounded_followup"

def get_conversation_mode(scenario: dict) -> str:
    return (scenario.get("conversation_mode") or "static").strip().lower()


def get_customer_turn_limits(scenario: dict) -> tuple[int, int]:
    ct = scenario.get("customer_turns", {}) or {}
    return int(ct.get("min", 1)), int(ct.get("max", 1))


def build_follow_up_plan_from_scenario(scenario: dict, outcome: dict) -> dict:
    conversation_mode = get_conversation_mode(scenario)
    _, max_turns = get_customer_turn_limits(scenario)

    if conversation_mode != "runtime_grounded":
        return {
            "mode": "none",
            "intent": "none",
            "trigger": "never",
            "outcome_manifest_stage": "initial_turn",
            "max_additional_turns": 0,
            "grounding_requirements": [],
        }

    gc = scenario.get("generation_constraints", {}) or {}
    grounding = gc_list(gc, "follow_up_grounding")

    stage = infer_outcome_manifest_stage(scenario, outcome)
    intent = infer_follow_up_intent(scenario, outcome)

    if intent == "none":
        return {
            "mode": "none",
            "intent": "none",
            "trigger": "never",
            "outcome_manifest_stage": stage,
            "max_additional_turns": 0,
            "grounding_requirements": [],
        }

    if stage == "initial_turn":
        trigger = "only_if_soa_requests_customer_followup"
    elif stage == "follow_up_turn":
        trigger = "after_existing_context_is_established"
    else:
        trigger = "context_dependent"

    return {
        "mode": "runtime_grounded",
        "intent": intent,
        "trigger": trigger,
        "outcome_manifest_stage": stage,
        "max_additional_turns": max(0, max_turns - 1),
        "grounding_requirements": grounding,
    }


def build_persona_index(personas: list) -> dict:
    return {p["persona_id"]: p for p in personas}


def build_tone_index(tones: list) -> dict:
    return {t["tone_id"]: t for t in tones}


def load_domain_scenarios(domain: str) -> list:
    path = os.path.join(SPACE_DIR, f"{domain}.yaml")
    if not os.path.exists(path):
        return []
    data = load_yaml(path)
    return data.get("scenarios", [])


def find_scenario_and_outcome(domain: str, scenario_id: str, outcome_id: str):
    scenarios = load_domain_scenarios(domain)
    for scenario in scenarios:
        if scenario.get("scenario_id") != scenario_id:
            continue
        for outcome in scenario.get("variants", []):
            if outcome.get("outcome_id") == outcome_id:
                return scenario, outcome
    return None, None


def extract_latest_soa_reply(messages: list) -> Optional[str]:

    if not messages:
        return None

    def get_content(msg):
        return (
            msg.get("messageContent")
            or msg.get("content")
            or msg.get("body")
            or ""
        )

    def get_author_text(msg):
        author_bits = [
            str(msg.get("from") or ""),
            str(msg.get("fromText") or ""),
            str(msg.get("author") or ""),
            str(msg.get("sender") or ""),
            str(msg.get("role") or ""),
        ]
        return " ".join(author_bits).lower()

    def get_type(msg):
        return str(msg.get("type") or "").lower()

    for msg in reversed(messages):
        content = str(get_content(msg)).strip()
        if not content:
            continue

        author_text = get_author_text(msg)
        msg_type = get_type(msg)

        if any(k in author_text for k in ["sales order agent", "soa", "assistant"]):
            return content

        if msg_type in {"assistant", "reply", "output"}:
            if len(content) >= 20:
                return content

    return None

# -----------------------------
# Sender selection
# -----------------------------
def map_sender_type_to_mode(sender_type: str) -> str:
    mapping = {
        "recognized_valid_customer": "valid",
        "unknown_but_plausible_sender": "invalid",
        "suspicious_sender": "suspicious",
    }
    return mapping.get(sender_type, "valid")


def determine_sender_mode(scenario: dict, outcome: dict) -> str:

    sender_constraints = scenario.get("sender_constraints", {}) or {}

    required_sender_type = sender_constraints.get("required_sender_type")
    if required_sender_type:
        return map_sender_type_to_mode(required_sender_type)

    allowed_sender_types = sender_constraints.get("allowed_sender_types", [])
    if allowed_sender_types:
        # Prefer valid if allowed
        if "recognized_valid_customer" in allowed_sender_types:
            return "valid"
        return map_sender_type_to_mode(allowed_sender_types[0])

    scenario_id = (scenario.get("scenario_id") or "").lower()
    outcome_id = (outcome.get("outcome_id") or "").lower()
    gc = scenario.get("generation_constraints", {}) or {}
    request_patterns = [p.lower() for p in gc_list(gc, "request_patterns")]

    # Explicit unknown customer scenarios
    if "unknown_customer" in scenario_id:
        return "invalid"
    if "quote_request_from_unknown_sender" in request_patterns:
        return "invalid"
    if "sender_not_resolved_to_existing_customer" in request_patterns:
        return "invalid"

    # Suspicious/phishing/prompt-injection style guardrails (future-facing)
    suspicious_signals = {
        "suspicious_sender",
        "prompt_injection_or_instruction_to_avoid_annotation",
    }
    if any(p in suspicious_signals for p in request_patterns):
        return "suspicious"

    # default
    return "valid"

async def generate_runtime_followup(
    kernel: Kernel,
    scenario: dict,
    outcome: dict,
    persona: dict,
    tone: dict,
    selected_sender: dict,
    follow_up_plan: dict,
    previous_soa_reply: str,
    thread_context: str,
    original_customer_request: str = "",
) -> Turn:
    safe_scenario, safe_outcome = build_customer_agent_view(scenario, outcome)

    prompt = FOLLOWUP_AGENT_PROMPT
    prompt += "\n\nSCENARIO:\n" + yaml.dump(safe_scenario, allow_unicode=True, sort_keys=False)
    prompt += "\nSELECTED OUTCOME:\n" + yaml.dump(safe_outcome, allow_unicode=True, sort_keys=False)
    prompt += "\nSELECTED PERSONA:\n" + build_persona_prompt(persona)
    prompt += "\nSELECTED TONE:\n" + build_tone_prompt(tone)
    prompt += "\nSELECTED SENDER:\n" + yaml.dump(selected_sender, allow_unicode=True, sort_keys=False)
    prompt += "\nFOLLOW-UP PLAN:\n" + yaml.dump(follow_up_plan, allow_unicode=True, sort_keys=False)
    prompt += "\nCONVERSATION THREAD:\n" + (thread_context or "")
    prompt += "\nLATEST SOA REPLY:\n" + (previous_soa_reply or "")
    prompt += "\nORIGINAL CUSTOMER REQUEST:\n" + (original_customer_request or "")

    last_error = None
    last_raw_output = None

    for attempt in range(5):
        try:
            raw_output = await call_model(kernel, prompt)
        except Exception as e:
            last_error = e
            print(f"[regen followup] model call failed on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 + attempt)
            continue

        last_raw_output = raw_output
        cleaned = strip_code_fences(raw_output)

        try:
            parsed = GeneratedTurn.model_validate_json(cleaned)

            subject = (parsed.subject or "").strip()
            body = enforce_sender_signature(parsed.body or "", selected_sender["name"])

            if is_bad_clarification_followup(body, follow_up_plan):
                print(f"[regen followup] bad clarification follow-up, retrying... attempt={attempt + 1}")
                continue

            if looks_like_low_information_followup(body, previous_soa_reply):
                print(f"[regen followup] low-information reply, retrying... attempt={attempt + 1}")
                continue

            return Turn(
                from_=selected_sender["email"],
                subject=subject,
                body=body,
            )

        except Exception as e:
            last_error = e
            print(f"[regen followup] parse failed on attempt {attempt + 1}: {e}")

    print("Follow-up generation failed after retries")
    if last_error:
        print("Last error:", last_error)
    if last_raw_output:
        print("Raw follow-up model output:")
        print(last_raw_output)

    # Hard fallback:
    # For clarification replies, never ask SOA to clarify again.
    # Instead choose one SOA option if possible.
    if (follow_up_plan.get("intent") or "").lower() == "reply_to_soa_clarification":
        print("[followup fallback] using deterministic clarification answer")
        return build_clarification_answer_fallback(
            previous_soa_reply=previous_soa_reply,
            selected_sender=selected_sender,
            follow_up_plan=follow_up_plan,
            original_customer_request=original_customer_request,
        )

    fallback_body = enforce_sender_signature(
        "Please proceed with the option you recommended.",
        selected_sender["name"],
    )

    return Turn(
        from_=selected_sender["email"],
        subject="Re:",
        body=fallback_body,
    )

async def generate_single_test_case(
    kernel: Kernel,
    scenario: dict,
    outcome: dict,
    persona: dict,
    tone: dict,
    items: list,
    selected_sender: dict,
    seed_key: str,
) -> Optional[GeneratedTestCase]:
    # Prepare prompt items using deterministic shuffle
    # so the model does not always see the same front-loaded items
    # -----------------------------
    prompt_items = prepare_items_for_prompt(items, seed_key=seed_key, max_items=54)
    items_summary = format_items_for_prompt(prompt_items, max_items=54)

    scenario_hints = build_scenario_hints(scenario, outcome)

    safe_scenario, safe_outcome = build_customer_agent_view(scenario, outcome)
    follow_up_plan = build_follow_up_plan_from_scenario(scenario, outcome)
    outcome_specific_hints = build_outcome_specific_hints(scenario, outcome)

    prompt = build_customer_agent_prompt_for_scenario(scenario)
    prompt += "\n\nSCENARIO:\n" + yaml.dump(safe_scenario, allow_unicode=True, sort_keys=False)
    prompt += "\nSELECTED OUTCOME:\n" + yaml.dump(safe_outcome, allow_unicode=True, sort_keys=False)
    prompt += "\nSELECTED PERSONA:\n" + build_persona_prompt(persona)
    prompt += "\nSELECTED TONE:\n" + build_tone_prompt(tone)
    prompt += "\nAVAILABLE ITEMS:\n" + items_summary
    prompt += "\nSELECTED SENDER:\n" + yaml.dump(selected_sender, allow_unicode=True, sort_keys=False)
    prompt += "\nSCENARIO HINTS:\n" + scenario_hints
    prompt += "\nEXPECTED FOLLOW-UP PLAN:\n" + yaml.dump(follow_up_plan, allow_unicode=True, sort_keys=False)
    prompt += "\nOUTCOME-SPECIFIC HINTS:\n" + outcome_specific_hints

    expected_data = dict(outcome.get("expected", {}))
    expected_data["sender"] = {
        "name": selected_sender["name"],
        "email": selected_sender["email"],
        "sender_type": selected_sender["sender_type"],
    }

    last_raw_output = None
    parsed = None
    last_error = None

    for attempt in range(3):
        try:
            raw_output = await call_model(kernel, prompt)
        except Exception as e:
            last_error = e
            print(f"[regen] model call failed on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 + attempt)
            continue

        last_raw_output = raw_output
        cleaned = strip_code_fences(raw_output)

        try:
            candidate = GeneratedCase.model_validate_json(cleaned)

            candidate_initial_text = (
                (candidate.initial_turn.subject or "") + "\n\n" + (candidate.initial_turn.body or "")
            )

            if not has_real_clarification_gap(scenario, outcome, candidate_initial_text, items):
                print(f"[regen] clarification gap not strong enough, retrying... attempt={attempt + 1}")
                continue

            if not has_real_intervention_trigger(scenario, outcome, candidate_initial_text, items):
                print(f"[regen] intervention trigger not strong enough, retrying... attempt={attempt + 1}")
                continue

            parsed = candidate
            break

        except Exception as e:
            last_error = e
            print(f"[regen] parse failed on attempt {attempt + 1}: {e}")

    if parsed is None:
        print("Parse/validation failed after retries")
        if last_error:
            print("Last error:", last_error)
        print("Raw model output:")
        print(last_raw_output)
        return None

    final_body = enforce_sender_signature(parsed.initial_turn.body or "", selected_sender["name"])

    initial_turn = Turn(
        from_=selected_sender["email"],
        subject=(parsed.initial_turn.subject or "").strip(),
        body=final_body,
    )

    return GeneratedTestCase(
        name=parsed.name,
        description=parsed.description,
        sender_name=selected_sender["name"],
        sender_email=selected_sender["email"],
        sender_type=selected_sender["sender_type"],
        persona_id=persona["persona_id"],
        persona_display_name=persona["display_name"],
        tone_id=tone["tone_id"],
        tone_display_name=tone["display_name"],
        initial_turn=initial_turn,
        follow_up_plan=follow_up_plan,
        expected_data=expected_data,
        attachments=[],
    )

async def wait_for_task_attention_or_stable(
    client: AgentTaskClient,
    task_id: int,
    max_wait_seconds: int = 120,
    poll_seconds: int = 2,
):
    start = time.time()
    last_task = None

    while time.time() - start < max_wait_seconds:
        elapsed = int(time.time() - start)

        task = client.get_task(task_id)
        last_task = task

        messages = client.get_messages(task_id)

        print(
            f"[poll] t={elapsed}s "
            f"task_id={task_id} "
            f"status={task.get('status')} "
            f"needsAttention={task.get('needsAttention')} "
            f"messages={len(messages)}"
        )

        if messages:
            last_msg = messages[-1]
            print(
                f"       last_message_type={last_msg.get('type')} "
                f"last_message_status={last_msg.get('status')} "
                f"content={str(last_msg.get('messageContent', ''))[:120]}"
            )

        if task.get("needsAttention"):
            print("[poll] -> needs_attention")
            return {"state": "needs_attention", "task": task}

        status = (task.get("status") or "").lower()

        if status in {"completed", "failed", "canceled"}:
            print("[poll] -> terminal")
            return {"state": "terminal", "task": task}

        if status == "paused":
            print("[poll] -> waiting_for_customer")
            return {"state": "waiting_for_customer", "task": task}

        await asyncio.sleep(poll_seconds)

    print("[poll] -> timeout")
    return {"state": "timeout", "task": last_task}

async def run_reviewer_until_pause_or_terminal(client: AgentTaskClient, task_id: int):
    from reviewer_agent import review_one_attention_step

    while True:
        task = client.get_task(task_id)
        print(
            f"[reviewer-loop] task_id={task_id}, "
            f"status={task.get('status')}, "
            f"needsAttention={task.get('needsAttention')}"
        )

        if not task.get("needsAttention"):
            return {"state": "no_attention", "task": task}

        print(f"[reviewer-loop] fetching intervention details for task_id={task_id}")

        decision = await review_one_attention_step(client, task_id)

        print(f"[reviewer-loop] decision={decision}")

        if decision["action"] == "stop":
            return {"state": "stopped", "task": task, "decision": decision}

        outcome = await wait_for_task_attention_or_stable(client, task_id)

        print(f"[reviewer-loop] post-approve outcome={outcome['state']}")

        if outcome["state"] == "needs_attention":
            continue

        return outcome

def normalize_message_newlines(text: str) -> str:
    """
    Normalize line endings for systems that expect CRLF.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "\r\n")

def build_thread_context(messages: list, max_messages: int = 8) -> str:

    if not messages:
        return ""

    selected = messages[-max_messages:]
    rendered = []

    for i, msg in enumerate(selected, start=1):
        content = (
            msg.get("messageContent")
            or msg.get("content")
            or msg.get("body")
            or ""
        )
        content = str(content).strip()
        if not content:
            continue

        author = (
            msg.get("from")
            or msg.get("fromText")
            or msg.get("author")
            or msg.get("sender")
            or msg.get("role")
            or "unknown"
        )
        author = str(author).strip()

        msg_type = str(msg.get("type") or "").strip()

        rendered.append(
            f"[{i}] author={author} type={msg_type}\n{content}"
        )

    return "\n\n".join(rendered)


async def run_generation_phase(
    kernel: Kernel,
    client: AgentTaskClient,
    tasks_input_dir: str
):
    print("Fetching available items from BC...")

    raw_items = client.get_items()
    items_all = [normalize_item_schema(x) for x in (raw_items or []) if x]

    print(f"[debug] raw_items={len(raw_items or [])}")

    print("[debug] first raw item:")
    if raw_items:
        print(json.dumps(raw_items[0], indent=2, ensure_ascii=False)[:3000])
    else:
        print("<NONE>")

    print("[debug] first normalized items:")
    for x in items_all[:5]:
        print(
            f"  number={x.get('number')} | "
            f"name={x.get('displayName')} | "
            f"uom={x.get('baseUnitOfMeasureCode')}"
        )

    # Case-insensitive benchmark item filter.
    # We only want items created by SOA-SETUP.yaml, e.g. AItem-0001.
    items = [
        i for i in items_all
        if str(i.get("number", "")).lower().startswith("aitem-")
    ]

    print(f"[debug] benchmark_items={len(items)}")
    print("[debug] first benchmark items:")
    for x in items[:5]:
        print(
            f"  {x.get('number')} | "
            f"{x.get('displayName')} | "
            f"{x.get('unitPrice')} | "
            f"{x.get('baseUnitOfMeasureCode')}"
        )

    if not items:
        raise ValueError(
            "No AItem-* items found after normalization. "
            "Check the raw item fields printed above and verify --company is the seeded company."
        )


    print("Fetching Customers from BC...")
    valid_customers = client.get_customers()

    print(f"[debug] valid customer senders={len(valid_customers)}")
    print("[debug] first valid customer senders:")
    for c in valid_customers[:10]:
        print(
            f"  number={c.get('number', '')} | "
            f"name={c.get('name', '')} | "
            f"email={c.get('email', '')} | "
            f"sender_type={c.get('sender_type', '')}"
        )

    if not valid_customers:
        raise ValueError(
            "No valid customers found in BC customers. "
            "For quote_created scenarios, sender must resolve to a real Sell-to Customer."
        )

    print("Loading personas and tones...")
    personas = load_personas(os.path.join(SPACE_DIR, "personas.yaml"))
    if not personas:
        raise ValueError("No personas found in space/personas.yaml.")

    tones = load_tones(os.path.join(SPACE_DIR, "tones.yaml"))
    if not tones:
        raise ValueError("No tones found in space/tones.yaml.")

    invalid_senders = build_invalid_senders(count=20)
    suspicious_senders = build_suspicious_senders(count=10)

    domain_files = [
        ("quote", os.path.join(SPACE_DIR, "quote.yaml"), 1),
        ("inquiry", os.path.join(SPACE_DIR, "inquiry.yaml"), 1),
        ("annotation", os.path.join(SPACE_DIR, "annotation.yaml"), 1),
        ("guardrail", os.path.join(SPACE_DIR, "guardrail.yaml"), 1),
    ]

    for domain, scenario_file, samples_per_variant in domain_files:
        print(f"\n=== GENERATING Domain: {domain} ===")

        if not os.path.exists(scenario_file):
            print(f"⚠ Skipping missing file: {scenario_file}")
            continue

        loaded = load_yaml(scenario_file)
        scenarios = loaded.get("scenarios", [])

        for scenario in scenarios:
            variants = scenario.get("variants", [])
            if not variants:
                print(f"⚠ Scenario has no variants: {scenario.get('scenario_id')}")
                continue

            for outcome in variants:
                scenario_id = scenario.get("scenario_id")
                outcome_id = outcome.get("outcome_id")
                print(f"Scenario: {scenario_id} | Outcome: {outcome_id}")

                input_payload = {
                    "scenario_id": scenario_id,
                    "outcome_id": outcome_id,
                    "test_cases": [],
                }

                oracle_payload = {
                    "scenario_id": scenario_id,
                    "outcome_id": outcome_id,
                    "test_cases": [],
                }

                for persona in personas:
                    persona_id = persona["persona_id"]
                    persona_display_name = persona.get("display_name", persona_id)

                    for tone in tones:
                        tone_id = tone["tone_id"]
                        tone_display_name = tone.get("display_name", tone_id)

                        for sample_idx in range(1, samples_per_variant + 1):
                            sender_mode = determine_sender_mode(scenario, outcome)

                            seed_key = (
                                f"{domain}:{scenario_id}:{outcome_id}:"
                                f"{persona_id}:{tone_id}:{sample_idx}:{sender_mode}"
                            )

                            selected_sender = choose_sender_for_case(
                                valid_customers=valid_customers,
                                invalid_senders=invalid_senders,
                                suspicious_senders=suspicious_senders,
                                sender_mode=sender_mode,
                                seed_key=seed_key,
                            )

                            try:
                                generated = await generate_single_test_case(
                                    kernel=kernel,
                                    scenario=scenario,
                                    outcome=outcome,
                                    persona=persona,
                                    tone=tone,
                                    items=items,
                                    selected_sender=selected_sender,
                                    seed_key=seed_key,
                                )
                            except Exception as e:
                                print(
                                    f"⚠ Generation crashed for case: "
                                    f"{scenario_id} / {outcome_id} / {persona_id} / {tone_id} / sample={sample_idx}"
                                )
                                print(f"   error={e}")
                                generated = None

                            if generated is None:
                                print(
                                    f"⚠ Skipping invalid generated case: "
                                    f"{scenario_id} / {outcome_id} / {persona_id} / {tone_id} / sample={sample_idx}"
                                )
                                continue

                            case_id = (
                                f"{scenario_id}__{outcome_id}__"
                                f"{persona_id}__{tone_id}__{sample_idx:03d}"
                            )

                            attachments = []

                            if (
                                scenario_requires_alternate_followup_sender(scenario)
                                and generated.follow_up_plan.get("mode") == "runtime_grounded"
                            ):
                                alternate_sender = choose_alternate_sender(
                                    valid_customers=valid_customers,
                                    primary_sender=selected_sender,
                                    seed_key=seed_key,
                                )
                                generated.follow_up_plan["follow_up_sender_name"] = alternate_sender["name"]
                                generated.follow_up_plan["follow_up_sender_email"] = alternate_sender["email"]
                                generated.follow_up_plan["follow_up_sender_type"] = alternate_sender["sender_type"]

                            input_payload["test_cases"].append({
                                "case_id": case_id,
                                "persona_id": persona_id,
                                "persona_display_name": persona_display_name,
                                "tone_id": tone_id,
                                "tone_display_name": tone_display_name,
                                "name": generated.name,
                                "description": generated.description,
                                "sender_name": generated.sender_name,
                                "sender_email": generated.sender_email,
                                "sender_type": generated.sender_type,
                                "initial_turn": generated.initial_turn.model_dump(),
                                "follow_up_plan": generated.follow_up_plan,
                            })

                            oracle_payload["test_cases"].append({
                                "case_id": case_id,
                                "persona_id": persona_id,
                                "persona_display_name": persona_display_name,
                                "tone_id": tone_id,
                                "tone_display_name": tone_display_name,
                                "name": generated.name,
                                "expected_data": generated.expected_data,
                            })

                input_save_path = os.path.join(
                    tasks_input_dir, domain, f"{scenario_id}__{outcome_id}.json"
                )

                os.makedirs(os.path.dirname(input_save_path), exist_ok=True)

                with open(input_save_path, "w", encoding="utf-8") as f:
                    json.dump(input_payload, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved input → {input_save_path}")


async def run_execution_phase(kernel: Kernel, client: AgentTaskClient, tasks_input_dir: str):
    task_files = []

    personas = load_personas(os.path.join(SPACE_DIR, "personas.yaml"))
    tones = load_tones(os.path.join(SPACE_DIR, "tones.yaml"))
    persona_index = build_persona_index(personas)
    tone_index = build_tone_index(tones)

    for root, _, files in os.walk(tasks_input_dir):
        for f in files:
            if f.endswith(".json"):
                task_files.append(os.path.join(root, f))

    print(f"Found {len(task_files)} task input files")

    for task_file in task_files:
        print(f"\n📂 Executing: {task_file}")

        with open(task_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        domain = os.path.basename(os.path.dirname(task_file))
        scenario_id = data.get("scenario_id")
        outcome_id = data.get("outcome_id")
        scenario, outcome = find_scenario_and_outcome(domain, scenario_id, outcome_id)

        if not scenario or not outcome:
            print(f"⚠ Could not resolve scenario/outcome for file: {task_file}")
            continue

        for test_case in data.get("test_cases", []):
            case_id = test_case.get("case_id", test_case.get("name", "unknown_case"))
            print(f"\n=== Test Case: {test_case['name']} ===")
            print(f"case_id={case_id}")

            initial_turn = test_case.get("initial_turn")
            follow_up_plan = test_case.get("follow_up_plan", {"mode": "none", "max_additional_turns": 0, "grounding_requirements": []})

            # backward compatibility for old generated files
            legacy_turns = test_case.get("turns", [])
            if not initial_turn and legacy_turns:
                initial_turn = legacy_turns[0]

            if not initial_turn:
                print("⚠ No initial_turn generated, skipping.")
                continue

            original_customer_request = build_initial_email_message(initial_turn, None)

            persona = persona_index.get(test_case.get("persona_id"))
            tone = tone_index.get(test_case.get("tone_id"))

            if not persona or not tone:
                print("⚠ Missing persona or tone definition, skipping.")
                continue

            selected_sender = {
                "name": test_case["sender_name"],
                "email": test_case["sender_email"],
                "sender_type": test_case["sender_type"],
            }

            task_id = None

            try:
                # --------------------------------------------------
                # Initial customer turn
                # --------------------------------------------------
                sender = initial_turn.get("from") or initial_turn.get("from_")
                msg = build_initial_email_message(initial_turn, None)

                print("🆕 Creating task with initial customer message...")

                result = client.create_task_and_wait(
                    from_text=sender,
                    message_text=msg,
                    external_id=case_id,
                )

                task_id = result.get("taskId")
                print(
                    f"   task_id={task_id}, "
                    f"status={result.get('status')}, "
                    f"needsAttention={result.get('needsAttention')}"
                )

                if not task_id:
                    print("❌ create_task_and_wait did not return taskId.")
                    continue

                task = client.get_task(task_id)
                status = (task.get("status") or "").lower()

                if task.get("needsAttention"):
                    print(f"🛑 task_id={task_id} entered Needs Attention after first turn.")
                    outcome_state = await run_reviewer_until_pause_or_terminal(client, task_id)
                elif status in {"completed", "failed", "canceled"}:
                    print("✅ Task reached terminal state.")
                    continue
                else:
                    outcome_state = await wait_for_task_attention_or_stable(
                        client,
                        task_id,
                        max_wait_seconds=30,
                        poll_seconds=2,
                    )

                if outcome_state["state"] == "stopped":
                    print("⏹ Reviewer stopped this case.")
                    continue

                if outcome_state["state"] == "terminal":
                    print("✅ Task reached terminal state.")
                    continue

                # --------------------------------------------------
                # Runtime grounded follow-up(s)
                # --------------------------------------------------
                remaining_followups = int(follow_up_plan.get("max_additional_turns", 0))
                mode = (follow_up_plan.get("mode") or "none").lower()

                # old files fallback: if legacy turns has more turns, replay them
                if mode == "none" and len(legacy_turns) > 1:
                    for turn in legacy_turns[1:]:
                        sender = turn.get("from") or turn.get("from_")
                        msg = build_followup_message(turn)

                        print(f"📨 Adding legacy follow-up customer message to task_id={task_id}...")

                        try:
                            result = client.add_message(
                                task_id,
                                from_text=sender,
                                message_text=msg,
                            )
                            print(
                                f"   completed={result.get('completed')}, "
                                f"status={result.get('status')}, "
                                f"needsAttention={result.get('needsAttention')}"
                            )
                        except requests.exceptions.HTTPError as e:
                            if e.response is not None and e.response.status_code == 408:
                                print(f"⚠ AddMessage timed out for task_id={task_id}.")
                            else:
                                raise
                        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
                            print(f"⚠ AddMessage client-side timeout for task_id={task_id}.")

                        outcome_state = await wait_for_task_attention_or_stable(client, task_id)

                        if outcome_state["state"] == "needs_attention":
                            print(f"🛑 task_id={task_id} entered Needs Attention.")
                            outcome_state = await run_reviewer_until_pause_or_terminal(client, task_id)

                        if outcome_state["state"] in {"stopped", "terminal", "timeout"}:
                            break

                    continue

                if mode != "runtime_grounded":
                    print("✅ No runtime-grounded follow-up required.")
                    continue

                while remaining_followups > 0:
                    if outcome_state["state"] == "needs_attention":
                        outcome_state = await run_reviewer_until_pause_or_terminal(client, task_id)

                    if outcome_state["state"] == "stopped":
                        print("⏹ Reviewer stopped this case.")
                        break

                    if outcome_state["state"] == "terminal":
                        print("✅ Task reached terminal state.")
                        break

                    if outcome_state["state"] == "timeout":
                        print("⏳ Task polling timed out.")
                        break

                    if outcome_state["state"] != "waiting_for_customer":
                        print(f"⚠ Unexpected state before runtime follow-up: {outcome_state['state']}")
                        break

                    messages = client.get_messages(task_id)
                    previous_soa_reply = extract_latest_soa_reply(messages)
                    thread_context = build_thread_context(messages)

                    print("[debug] extracted previous_soa_reply:")
                    print(previous_soa_reply or "<NONE>")

                    print("[debug] thread_context:")
                    print(thread_context[:2000])

                    if not previous_soa_reply:
                        print("⚠ Could not extract SOA reply for grounded follow-up.")
                        break

                    runtime_sender = selected_sender

                    if follow_up_plan.get("follow_up_sender_email"):
                        runtime_sender = {
                            "name": follow_up_plan.get("follow_up_sender_name", selected_sender["name"]),
                            "email": follow_up_plan.get("follow_up_sender_email", selected_sender["email"]),
                            "sender_type": follow_up_plan.get("follow_up_sender_type", selected_sender["sender_type"]),
                        }

                    followup_turn = await generate_runtime_followup(
                        kernel=kernel,
                        scenario=scenario,
                        outcome=outcome,
                        persona=persona,
                        tone=tone,
                        selected_sender=runtime_sender,
                        follow_up_plan=follow_up_plan,
                        previous_soa_reply=previous_soa_reply,
                        thread_context=thread_context,
                        original_customer_request=original_customer_request,
                    )

                    print(f"📨 Adding runtime-grounded follow-up to task_id={task_id}...")
                    print(f"   subject={followup_turn.subject}")

                    try:
                        result = client.add_message(
                            task_id,
                            from_text=followup_turn.from_,
                            message_text=build_followup_message(followup_turn.model_dump()),
                        )
                        print(
                            f"   completed={result.get('completed')}, "
                            f"status={result.get('status')}, "
                            f"needsAttention={result.get('needsAttention')}"
                        )
                    except requests.exceptions.HTTPError as e:
                        if e.response is not None and e.response.status_code == 408:
                            print(f"⚠ AddMessage timed out for task_id={task_id}.")
                        else:
                            raise
                    except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
                        print(f"⚠ AddMessage client-side timeout for task_id={task_id}.")

                    remaining_followups -= 1
                    outcome_state = await wait_for_task_attention_or_stable(client, task_id)

                if outcome_state["state"] == "terminal":
                    print("✅ Task reached terminal state.")
                elif outcome_state["state"] == "waiting_for_customer":
                    print("⏸ SOA is waiting for customer, but no more follow-ups are allowed in this scenario.")
                elif outcome_state["state"] == "timeout":
                    print("⏳ Task polling timed out.")

            except Exception as e:
                print(f"❌ Failed case: {case_id}")
                print(str(e))
                continue

# -----------------------------
# Main
# -----------------------------
async def main():
    parser = argparse.ArgumentParser(description="Customer Agent Runner")

    parser.add_argument("--mode", required=True, choices=["generate", "execute"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--company", default="")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--tasks-input-dir", default="datasets")

    args = parser.parse_args()

    client = AgentTaskClient(
        base_url=args.base_url,
        company=args.company,
        user=args.user,
        password=args.password,
    )

    if args.mode == "generate":
        kernel = build_kernel()
        await run_generation_phase(
            kernel,
            client,
            tasks_input_dir=args.tasks_input_dir
        )

    elif args.mode == "execute":
        kernel = build_kernel()
        await run_execution_phase(kernel, client, tasks_input_dir=args.tasks_input_dir)

if __name__ == "__main__":
    asyncio.run(main())