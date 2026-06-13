# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import asyncio
import json
import re
import html
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings


# =========================================================
# 1. Schema enums
# =========================================================

TURN_BEHAVIOR_ENUM = [
    "create_quote",
    "update_quote",
    "create_sales_order",
    "update_sales_order",
    "convert_quote_to_sales_order",
    "ask_clarification",
    "safe_refusal",
    "intervention",
    "annotation",
    "availability_response",
    "inquiry_response",
    "no_action",
    "other",
]

DOCUMENT_TYPE_ENUM = [
    "sales_quote",
    "sales_order",
    "availability_response",
    "clarification_request",
    "intervention_required",
    "annotation",
    "inquiry_response",
    "no_document",
    "unknown",
    "not_applicable",
]

STATUS_ENUM = [
    "ok",
    "partially_correct",
    "incorrect",
    "not_evaluable",
    "not_applicable",
    "unknown",
]

CONTEXT_USAGE_ENUM = [
    "standalone",
    "context_dependent",
]

ITEM_CHECKED_AGAINST_ENUM = [
    "customer_context_vs_soa_summary_with_bc_catalog",
    "soa_summary_only",
    "not_evaluable",
    "not_applicable",
    "unknown",
]


ITEM_OBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_text",
        "item_no",
        "description",
        "quantity",
        "uom",
        "modifiers",
        "confidence",
    ],
    "properties": {
        "source_text": {"type": "string"},
        "item_no": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "quantity": {"type": ["number", "string", "null"]},
        "uom": {"type": ["string", "null"]},
        "modifiers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "unknown"],
        },
    },
}


REQUESTED_ITEMS_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["turn_requested_items"],
    "properties": {
        "turn_requested_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "turn_index",
                    "requires_item_check",
                    "requested_items",
                    "reason",
                ],
                "properties": {
                    "turn_index": {"type": "integer"},
                    "requires_item_check": {"type": "boolean"},
                    "requested_items": {
                        "type": "array",
                        "items": ITEM_OBJECT_SCHEMA,
                    },
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


LLM_EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "failure_reason",
        "reason",
        "turn_evaluations",
        "final_outcome_validation",
        "summary_validation",
        "diagnostic_metrics",
        "availability_factuality",
        "trajectory_consistency",
    ],
    "properties": {
        "failure_reason": {
            "type": ["string", "null"],
        },
        "reason": {
            "type": "string",
        },
        "turn_evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "turn_index",
                    "context_usage",
                    "context_used_reason",
                    "customer_request_summary",
                    "soa_reply_summary",
                    "expected_turn_behavior",
                    "actual_turn_behavior",
                    "item_check",
                    "turn_success",
                    "failure_reason",
                    "reason",
                ],
                "properties": {
                    "turn_index": {"type": "integer"},
                    "context_usage": {
                        "type": "string",
                        "enum": CONTEXT_USAGE_ENUM,
                    },
                    "context_used_reason": {"type": "string"},
                    "customer_request_summary": {"type": "string"},
                    "soa_reply_summary": {"type": "string"},
                    "expected_turn_behavior": {
                        "type": "string",
                        "enum": TURN_BEHAVIOR_ENUM,
                    },
                    "actual_turn_behavior": {
                        "type": "string",
                        "enum": TURN_BEHAVIOR_ENUM,
                    },
                    "item_check": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "requires_item_check",
                            "status",
                            "checked_against",
                            "requested_items",
                            "actual_items",
                            "missing_items",
                            "wrong_items",
                            "wrong_quantity",
                            "wrong_uom",
                            "wrong_variant_or_modifier",
                            "extra_or_fabricated_items",
                            "reason",
                        ],
                        "properties": {
                            "requires_item_check": {"type": "boolean"},
                            "status": {
                                "type": "string",
                                "enum": STATUS_ENUM,
                            },
                            "checked_against": {
                                "type": "string",
                                "enum": ITEM_CHECKED_AGAINST_ENUM,
                            },
                            "requested_items": {
                                "type": "array",
                                "items": ITEM_OBJECT_SCHEMA,
                            },
                            "actual_items": {
                                "type": "array",
                                "items": ITEM_OBJECT_SCHEMA,
                            },
                            "missing_items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "wrong_items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "wrong_quantity": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "wrong_uom": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "wrong_variant_or_modifier": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "extra_or_fabricated_items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                    },
                    "turn_success": {"type": "boolean"},
                    "failure_reason": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "final_outcome_validation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "expected_outcome",
                "actual_outcome",
                "final_outcome_success",
                "reason",
            ],
            "properties": {
                "expected_outcome": {"type": "string"},
                "actual_outcome": {"type": "string"},
                "final_outcome_success": {"type": "boolean"},
                "reason": {"type": "string"},
            },
        },
        "summary_validation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "status",
                "summary_contains_correct_answer",
                "final_soa_text",
                "expected_items",
                "actual_items",
                "missing_items",
                "wrong_quantity",
                "extra_items",
                "document_type_expected",
                "document_type_claimed",
                "policy_refusal_detected",
                "sensitive_data_leak_detected",
            ],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": STATUS_ENUM,
                },
                "summary_contains_correct_answer": {"type": "boolean"},
                "final_soa_text": {"type": "string"},
                "expected_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "actual_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "missing_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "wrong_quantity": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "extra_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "document_type_expected": {
                    "type": "string",
                    "enum": DOCUMENT_TYPE_ENUM,
                },
                "document_type_claimed": {
                    "type": "string",
                    "enum": DOCUMENT_TYPE_ENUM,
                },
                "policy_refusal_detected": {"type": "boolean"},
                "sensitive_data_leak_detected": {"type": "boolean"},
            },
        },
        "diagnostic_metrics": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "item_correctness_status",
                "item_correctness_score",
                "document_type_accuracy",
                "availability_factuality_status",
                "availability_factuality_score",
                "trajectory_consistency_score",
            ],
            "properties": {
                "item_correctness_status": {
                    "type": "string",
                    "enum": STATUS_ENUM,
                },
                "item_correctness_score": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "document_type_accuracy": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "availability_factuality_status": {
                    "type": "string",
                    "enum": STATUS_ENUM,
                },
                "availability_factuality_score": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "trajectory_consistency_score": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
        },
        "availability_factuality": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "status",
                "availability_factuality_score",
                "availability_factuality_pass",
                "inconsistent_claims",
            ],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": STATUS_ENUM,
                },
                "availability_factuality_score": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "availability_factuality_pass": {"type": "boolean"},
                "inconsistent_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "trajectory_consistency": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "trajectory_consistency_score",
                "trajectory_consistent",
                "quantity_contradictions",
                "other_contradictions",
            ],
            "properties": {
                "trajectory_consistency_score": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "trajectory_consistent": {"type": "boolean"},
                "quantity_contradictions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "other_contradictions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
}


# =========================================================
# 2. Basic helpers
# =========================================================

def normalize_eval_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _text(msg: Optional[dict]) -> str:
    if not msg:
        return ""

    return str(
        msg.get("messageContent")
        or msg.get("content")
        or msg.get("body")
        or msg.get("text")
        or ""
    ).strip()


def _author(msg: dict) -> str:
    return str(
        msg.get("from")
        or msg.get("fromText")
        or msg.get("author")
        or msg.get("sender")
        or msg.get("role")
        or "unknown"
    ).strip()


def _role_text(msg: dict) -> str:
    return " ".join([
        str(msg.get("direction") or ""),
        str(msg.get("type") or ""),
        str(msg.get("from") or ""),
        str(msg.get("fromText") or ""),
        str(msg.get("author") or ""),
        str(msg.get("sender") or ""),
        str(msg.get("role") or ""),
    ]).lower()


def _is_soa_message(msg: dict) -> bool:
    role_text = _role_text(msg)
    direction = str(msg.get("direction") or "").lower()
    msg_type = str(msg.get("type") or "").lower()
    role = str(msg.get("role") or "").lower()

    return (
        direction == "output"
        or msg_type in {"output", "assistant", "reply"}
        or role == "assistant"
        or "sales order agent" in role_text
        or "soa" in role_text
    )


def _is_customer_message(msg: dict) -> bool:
    role_text = _role_text(msg)
    direction = str(msg.get("direction") or "").lower()
    msg_type = str(msg.get("type") or "").lower()
    role = str(msg.get("role") or "").lower()

    if _is_soa_message(msg):
        return False

    return (
        direction == "input"
        or msg_type in {"input", "user", "customer", "email"}
        or role in {"user", "customer"}
        or "customer" in role_text
    )


def build_conversation(messages: List[dict]) -> List[dict]:
    conversation = []

    for idx, msg in enumerate(messages or []):
        content = _text(msg)
        if not content:
            continue

        conversation.append({
            "index": idx,
            "from": _author(msg),
            "text": content,
            "normalized_text": normalize_eval_text(content),
            "is_soa": _is_soa_message(msg),
            "is_customer": _is_customer_message(msg),
            "raw_metadata": {
                "direction": msg.get("direction"),
                "type": msg.get("type"),
                "from": msg.get("from"),
                "fromText": msg.get("fromText"),
                "author": msg.get("author"),
                "sender": msg.get("sender"),
                "role": msg.get("role"),
            },
        })

    return conversation


def get_final_soa_message(messages: List[dict]) -> Optional[dict]:
    for msg in reversed(messages or []):
        if _is_soa_message(msg) and _text(msg):
            return msg
    return None


def build_customer_soa_turns(messages: List[dict]) -> List[dict]:
    conversation = build_conversation(messages)
    turns = []
    pending_customer = None

    for msg in conversation:
        if msg.get("is_customer"):
            if pending_customer is None:
                pending_customer = dict(msg)
            else:
                pending_customer["text"] += "\n" + msg["text"]
                pending_customer["normalized_text"] += "\n" + msg["normalized_text"]
            continue

        if msg.get("is_soa"):
            if pending_customer is not None:
                turns.append({
                    "turn_index": len(turns) + 1,
                    "customer_message_index": pending_customer["index"],
                    "soa_message_index": msg["index"],
                    "customer_text": pending_customer["normalized_text"],
                    "soa_text": msg["normalized_text"],
                    "customer_raw_text": pending_customer["text"],
                    "soa_raw_text": msg["text"],
                })
                pending_customer = None

    if pending_customer is not None:
        turns.append({
            "turn_index": len(turns) + 1,
            "customer_message_index": pending_customer["index"],
            "soa_message_index": None,
            "customer_text": pending_customer["normalized_text"],
            "soa_text": "",
            "customer_raw_text": pending_customer["text"],
            "soa_raw_text": "",
            "missing_soa_reply": True,
        })

    return turns


# =========================================================
# 3. BC catalog evidence
# =========================================================

def get_catalog_items(client) -> List[dict]:
    """
    Get normalized BC catalog items.

    Prefer client.get_items() if available.
    Fall back to client.list_items() if get_items is not implemented.
    """
    try:
        if hasattr(client, "get_items"):
            raw_items = client.get_items()
        elif hasattr(client, "list_items"):
            raw_items = client.list_items()
        else:
            raw_items = []
    except Exception:
        return []

    catalog = []
    seen = set()

    for item in raw_items or []:
        if not isinstance(item, dict):
            continue

        raw_item_no = str(
            item.get("itemNo")
            or item.get("number")
            or item.get("no")
            or item.get("No")
            or item.get("item_no")
            or ""
        ).strip()

        if not raw_item_no:
            continue

        item_no_l = raw_item_no.lower()
        if item_no_l in seen:
            continue

        seen.add(item_no_l)

        description = str(
            item.get("displayName")
            or item.get("description")
            or item.get("Description")
            or item.get("name")
            or item.get("Name")
            or ""
        ).strip()

        uom = str(
            item.get("baseUnitOfMeasureCode")
            or item.get("unitOfMeasure")
            or item.get("unitOfMeasureCode")
            or item.get("uom")
            or item.get("UOM")
            or ""
        ).strip()

        catalog.append({
            "item_no": item_no_l,
            "itemNo": raw_item_no,
            "description": description,
            "unitOfMeasure": uom,
        })

    return catalog


async def get_catalog_items_async(client) -> List[dict]:
    return await asyncio.to_thread(get_catalog_items, client)


def get_valid_item_nos_from_catalog(catalog_items: List[dict]) -> set:
    return set(
        str(i.get("item_no") or "").lower()
        for i in catalog_items
        if i.get("item_no")
    )


def get_valid_item_nos(client) -> set:
    try:
        if hasattr(client, "get_items"):
            items = client.get_items()
        elif hasattr(client, "list_items"):
            items = client.list_items()
        else:
            items = []

        return set(
            str(
                i.get("itemNo")
                or i.get("number")
                or i.get("no")
                or i.get("item_no")
                or ""
            ).lower()
            for i in items
            if i.get("itemNo") or i.get("number") or i.get("no") or i.get("item_no")
        )
    except Exception:
        return set()


async def get_valid_item_nos_async(client) -> set:
    return await asyncio.to_thread(get_valid_item_nos, client)


# =========================================================
# 4. Optional weak expected item evidence
# =========================================================

def _parse_qty(value: Any):
    if value is None:
        return None

    try:
        n = float(value)
        return int(n) if n.is_integer() else n
    except Exception:
        return value


def _first_present(d: dict, *keys: str):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def extract_optional_expected_items_from_case(
    case: dict,
    outcome: dict,
    valid_item_nos: set,
) -> List[dict]:
    candidates = []

    for source in [case, outcome]:
        if not isinstance(source, dict):
            continue

        for key in [
            "expected_items",
            "requested_items",
            "items",
            "order_lines",
            "quote_lines",
        ]:
            value = source.get(key)
            if isinstance(value, list):
                candidates.extend(value)

        expected_data = source.get("expected_data")
        if isinstance(expected_data, dict):
            for key in [
                "expected_items",
                "requested_items",
                "items",
                "order_lines",
                "quote_lines",
            ]:
                value = expected_data.get(key)
                if isinstance(value, list):
                    candidates.extend(value)

            for quote in expected_data.get("quotes", []) or []:
                if isinstance(quote, dict) and isinstance(quote.get("lines"), list):
                    candidates.extend(quote["lines"])

            for order in expected_data.get("salesOrders", []) or []:
                if isinstance(order, dict) and isinstance(order.get("lines"), list):
                    candidates.extend(order["lines"])

            for turn in expected_data.get("expectedTurns", []) or []:
                if isinstance(turn, dict) and isinstance(turn.get("expectedLines"), list):
                    candidates.extend(turn["expectedLines"])

    normalized = []

    for item in candidates:
        if isinstance(item, str):
            normalized.append({
                "description": item,
                "source": "optional_static_expected_items",
            })
            continue

        if not isinstance(item, dict):
            continue

        raw_item_no = str(
            item.get("itemNo")
            or item.get("item_no")
            or item.get("number")
            or item.get("no")
            or ""
        ).strip()

        item_no_l = raw_item_no.lower() if raw_item_no else ""

        description = str(
            item.get("name")
            or item.get("description")
            or item.get("displayName")
            or ""
        ).strip()

        qty = _first_present(item, "quantity", "qty")
        uom = _first_present(item, "unitOfMeasure", "uom", "unit", "unitOfMeasureCode")

        parsed = {
            "source": "optional_static_expected_items",
        }

        if raw_item_no:
            if valid_item_nos and item_no_l not in valid_item_nos:
                continue
            parsed["item_no"] = item_no_l
            parsed["itemNo"] = raw_item_no

        if description:
            parsed["description"] = description

        if qty is not None:
            parsed["quantity"] = _parse_qty(qty)

        if uom:
            parsed["uom"] = str(uom).strip()

        if "item_no" in parsed or "description" in parsed:
            normalized.append(parsed)

    return normalized


# =========================================================
# 5. Evidence builders
# =========================================================

def build_turn_item_evidence(turns: List[dict]) -> List[dict]:
    evidence = []

    for turn in turns or []:
        evidence.append({
            "turn_index": turn.get("turn_index"),
            "customer_message_index": turn.get("customer_message_index"),
            "soa_message_index": turn.get("soa_message_index"),
            "customer_text": turn.get("customer_text") or "",
            "soa_text": turn.get("soa_text") or "",
            "extraction_instruction": (
                "Decide semantically whether this turn requires item checking. "
                "If yes, extract requested_items from the customer request and necessary prior context. "
                "Extract actual_items only from SOA-visible summary/reply text. "
                "Do not infer hidden Business Central quote or sales order document contents."
            ),
        })

    return evidence


def analyze_soa_trajectory_for_evidence(messages: List[dict]) -> Dict[str, Any]:
    soa_steps = []

    for idx, msg in enumerate(messages or []):
        if not _is_soa_message(msg):
            continue

        text = _text(msg)
        if not text:
            continue

        soa_steps.append({
            "step_index": idx,
            "author": _author(msg),
            "text": text,
            "normalized_text": normalize_eval_text(text),
            "instruction": (
                "Extract actual generated/claimed item lines only from this SOA-visible text. "
                "Do not infer hidden quote/order document contents."
            ),
        })

    return {
        "soa_steps": soa_steps,
    }


# =========================================================
# 6. LLM JSON parsing
# =========================================================

def _strip_json_code_fence(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    return text


def _extract_json_object(text: str) -> str:
    text = _strip_json_code_fence(text)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def _parse_llm_json_or_fail(text: str) -> Dict[str, Any]:
    json_text = _extract_json_object(text)

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        return {
            "failure_reason": "llm_json_parse_failed",
            "reason": f"LLM evaluator did not return valid JSON: {str(e)}",
            "turn_evaluations": [],
            "final_outcome_validation": {
                "expected_outcome": "",
                "actual_outcome": "",
                "final_outcome_success": False,
                "reason": "LLM evaluator JSON parse failed",
            },
            "summary_validation": {},
            "diagnostic_metrics": {},
            "availability_factuality": {},
            "trajectory_consistency": {},
            "raw": str(text),
        }

    if not isinstance(parsed, dict):
        return {
            "failure_reason": "llm_json_not_object",
            "reason": "LLM evaluator returned JSON, but it was not an object.",
            "turn_evaluations": [],
            "final_outcome_validation": {
                "expected_outcome": "",
                "actual_outcome": "",
                "final_outcome_success": False,
                "reason": "LLM evaluator returned non-object JSON",
            },
            "summary_validation": {},
            "diagnostic_metrics": {},
            "availability_factuality": {},
            "trajectory_consistency": {},
            "raw": parsed,
        }

    return parsed


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if value < 0.0:
        return 0.0

    if value > 1.0:
        return 1.0

    return value


# =========================================================
# 7. Normalization
# =========================================================

def normalize_item_objects(value: Any) -> List[dict]:
    if not isinstance(value, list):
        return []

    normalized = []

    for item in value:
        if not isinstance(item, dict):
            continue

        modifiers = item.get("modifiers")
        if not isinstance(modifiers, list):
            modifiers = []

        confidence = str(item.get("confidence") or "unknown")
        if confidence not in {"high", "medium", "low", "unknown"}:
            confidence = "unknown"

        item_no = item.get("item_no")
        if isinstance(item_no, str):
            item_no = item_no.lower().strip() or None

        normalized.append({
            "source_text": str(item.get("source_text") or ""),
            "item_no": item_no,
            "description": item.get("description"),
            "quantity": item.get("quantity"),
            "uom": item.get("uom"),
            "modifiers": [str(x) for x in modifiers],
            "confidence": confidence,
        })

    return normalized


def normalize_item_check(raw_item_check: Any) -> Dict[str, Any]:
    if not isinstance(raw_item_check, dict):
        raw_item_check = {}

    status = str(raw_item_check.get("status") or "unknown")
    if status not in STATUS_ENUM:
        status = "unknown"

    checked_against = str(raw_item_check.get("checked_against") or "unknown")
    if checked_against not in ITEM_CHECKED_AGAINST_ENUM:
        checked_against = "unknown"

    def list_or_empty(key: str) -> List[str]:
        value = raw_item_check.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]
        return []

    requires_item_check = bool(raw_item_check.get("requires_item_check") is True)

    if not requires_item_check:
        status = "not_applicable"
        checked_against = "not_applicable"

    return {
        "requires_item_check": requires_item_check,
        "status": status,
        "checked_against": checked_against,
        "requested_items": normalize_item_objects(raw_item_check.get("requested_items")),
        "actual_items": normalize_item_objects(raw_item_check.get("actual_items")),
        "missing_items": list_or_empty("missing_items"),
        "wrong_items": list_or_empty("wrong_items"),
        "wrong_quantity": list_or_empty("wrong_quantity"),
        "wrong_uom": list_or_empty("wrong_uom"),
        "wrong_variant_or_modifier": list_or_empty("wrong_variant_or_modifier"),
        "extra_or_fabricated_items": list_or_empty("extra_or_fabricated_items"),
        "reason": str(raw_item_check.get("reason") or ""),
    }


def compute_item_score_from_turns(turns: List[dict]) -> Optional[float]:
    scores = []

    for t in turns or []:
        if not isinstance(t, dict):
            continue

        item_check = t.get("item_check") or {}
        if not isinstance(item_check, dict):
            continue

        if item_check.get("requires_item_check") is not True:
            continue

        status = item_check.get("status")

        if status == "ok":
            scores.append(1.0)
        elif status == "partially_correct":
            scores.append(0.5)
        elif status == "incorrect":
            scores.append(0.0)
        elif status == "not_evaluable":
            continue

    return mean(scores) if scores else None

def compute_expected_behavior_accuracy_from_turns(turns: List[dict]) -> Optional[float]:
    matched = 0
    total = 0

    for t in turns or []:
        if not isinstance(t, dict):
            continue

        expected_behavior = str(t.get("expected_turn_behavior") or "other")
        actual_behavior = str(t.get("actual_turn_behavior") or "other")

        # Exclude fallback turns where expected behavior is not explicitly defined
        if expected_behavior == "other":
            continue

        total += 1
        if expected_behavior == actual_behavior:
            matched += 1

    return (matched / total) if total else None

def infer_item_status_from_score(
    score: Optional[float],
    turns: List[dict],
    current_status: str,
) -> str:
    has_item_turn = any(
        isinstance(t, dict)
        and isinstance(t.get("item_check"), dict)
        and t["item_check"].get("requires_item_check") is True
        for t in turns or []
    )

    if not has_item_turn:
        return "not_applicable"

    if score is None:
        if current_status in STATUS_ENUM and current_status != "unknown":
            return current_status
        return "not_evaluable"

    if score == 1.0:
        return "ok"

    if score == 0.0:
        return "incorrect"

    return "partially_correct"


def normalize_llm_eval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {}

    result["evaluation_mode"] = "llm_primary_summary_visible_item_extraction"

    result.setdefault("failure_reason", None)
    result.setdefault("reason", "")
    result.setdefault("turn_evaluations", [])
    result.setdefault("final_outcome_validation", {})
    result.setdefault("summary_validation", {})
    result.setdefault("diagnostic_metrics", {})
    result.setdefault("availability_factuality", {})
    result.setdefault("trajectory_consistency", {})

    raw_turns = result.get("turn_evaluations", [])
    if not isinstance(raw_turns, list):
        raw_turns = []

    normalized_turns = []

    for idx, turn in enumerate(raw_turns, start=1):
        if not isinstance(turn, dict):
            continue

        context_usage = str(turn.get("context_usage") or "standalone")
        if context_usage not in CONTEXT_USAGE_ENUM:
            context_usage = "standalone"

        expected_turn_behavior = str(turn.get("expected_turn_behavior") or "other")
        if expected_turn_behavior not in TURN_BEHAVIOR_ENUM:
            expected_turn_behavior = "other"

        actual_turn_behavior = str(turn.get("actual_turn_behavior") or "other")
        if actual_turn_behavior not in TURN_BEHAVIOR_ENUM:
            actual_turn_behavior = "other"

        item_check = normalize_item_check(turn.get("item_check"))

        normalized_turns.append({
            "turn_index": int(turn.get("turn_index") or idx),
            "context_usage": context_usage,
            "context_used_reason": str(turn.get("context_used_reason") or ""),
            "customer_request_summary": str(turn.get("customer_request_summary") or ""),
            "soa_reply_summary": str(turn.get("soa_reply_summary") or ""),
            "expected_turn_behavior": expected_turn_behavior,
            "actual_turn_behavior": actual_turn_behavior,
            "item_check": item_check,
            "turn_success": bool(turn.get("turn_success") is True),
            "failure_reason": turn.get("failure_reason"),
            "reason": str(turn.get("reason") or ""),
        })

    result["turn_evaluations"] = normalized_turns

    attempted_turns = len(normalized_turns)
    successful_turns = sum(
        1 for t in normalized_turns
        if t.get("turn_success") is True
    )

    all_turns_success = attempted_turns > 0 and successful_turns == attempted_turns
    multi_turn_accuracy_contribution = (
        successful_turns / attempted_turns if attempted_turns else 0.0
    )

    expected_behavior_accuracy = compute_expected_behavior_accuracy_from_turns(normalized_turns)


    result["turn_metrics"] = {
        "attempted_turns": attempted_turns,
        "successful_turns": successful_turns,
        "multi_turn_accuracy_contribution": multi_turn_accuracy_contribution,
        "all_turns_success": all_turns_success,
    }

    fov = result.get("final_outcome_validation", {})
    if not isinstance(fov, dict):
        fov = {}

    fov = {
        "expected_outcome": str(fov.get("expected_outcome") or ""),
        "actual_outcome": str(fov.get("actual_outcome") or ""),
        "final_outcome_success": bool(fov.get("final_outcome_success") is True),
        "reason": str(fov.get("reason") or ""),
    }

    result["final_outcome_validation"] = fov
    result["final_outcome_success"] = fov["final_outcome_success"]

    result["scenario_success"] = all_turns_success
    result["success"] = all_turns_success

    if successful_turns < attempted_turns:
        first_failed = next(
            (t for t in normalized_turns if t.get("turn_success") is not True),
            None,
        )
        result["failure_reason"] = (
            (first_failed or {}).get("failure_reason")
            or result.get("failure_reason")
            or "turn_failed"
        )
    else:
        result["failure_reason"] = None

    summary = result.get("summary_validation", {})
    if not isinstance(summary, dict):
        summary = {}

    summary_status = str(summary.get("status") or "unknown")
    if summary_status not in STATUS_ENUM:
        summary_status = "unknown"

    document_type_expected = str(summary.get("document_type_expected") or "unknown")
    if document_type_expected not in DOCUMENT_TYPE_ENUM:
        document_type_expected = "unknown"

    document_type_claimed = str(summary.get("document_type_claimed") or "unknown")
    if document_type_claimed not in DOCUMENT_TYPE_ENUM:
        document_type_claimed = "unknown"

    result["summary_validation"] = {
        "status": summary_status,
        "summary_contains_correct_answer": bool(summary.get("summary_contains_correct_answer") is True),
        "final_soa_text": str(summary.get("final_soa_text") or ""),
        "expected_items": summary.get("expected_items") if isinstance(summary.get("expected_items"), list) else [],
        "actual_items": summary.get("actual_items") if isinstance(summary.get("actual_items"), list) else [],
        "missing_items": summary.get("missing_items") if isinstance(summary.get("missing_items"), list) else [],
        "wrong_quantity": summary.get("wrong_quantity") if isinstance(summary.get("wrong_quantity"), list) else [],
        "extra_items": summary.get("extra_items") if isinstance(summary.get("extra_items"), list) else [],
        "document_type_expected": document_type_expected,
        "document_type_claimed": document_type_claimed,
        "policy_refusal_detected": bool(summary.get("policy_refusal_detected") is True),
        "sensitive_data_leak_detected": bool(summary.get("sensitive_data_leak_detected") is True),
    }

    availability = result.get("availability_factuality", {})
    if not isinstance(availability, dict):
        availability = {}

    availability_status = str(availability.get("status") or "unknown")
    if availability_status not in STATUS_ENUM:
        availability_status = "unknown"

    availability_score = _coerce_float_or_none(
        availability.get("availability_factuality_score")
    )

    result["availability_factuality"] = {
        "status": availability_status,
        "availability_factuality_score": availability_score,
        "availability_factuality_pass": bool(availability.get("availability_factuality_pass") is True),
        "inconsistent_claims": availability.get("inconsistent_claims")
        if isinstance(availability.get("inconsistent_claims"), list)
        else [],
    }

    trajectory = result.get("trajectory_consistency", {})
    if not isinstance(trajectory, dict):
        trajectory = {}

    trajectory_score = _coerce_float_or_none(
        trajectory.get("trajectory_consistency_score")
    )

    result["trajectory_consistency"] = {
        "trajectory_consistency_score": trajectory_score,
        "trajectory_consistent": bool(trajectory.get("trajectory_consistent") is True),
        "quantity_contradictions": trajectory.get("quantity_contradictions")
        if isinstance(trajectory.get("quantity_contradictions"), list)
        else [],
        "other_contradictions": trajectory.get("other_contradictions")
        if isinstance(trajectory.get("other_contradictions"), list)
        else [],
    }

    diagnostic = result.get("diagnostic_metrics", {})
    if not isinstance(diagnostic, dict):
        diagnostic = {}

    item_score = _coerce_float_or_none(diagnostic.get("item_correctness_score"))
    if item_score is None:
        item_score = compute_item_score_from_turns(normalized_turns)

    item_status = str(diagnostic.get("item_correctness_status") or "unknown")
    if item_status not in STATUS_ENUM:
        item_status = "unknown"

    item_status = infer_item_status_from_score(
        item_score,
        normalized_turns,
        item_status,
    )

    document_type_accuracy = _coerce_float_or_none(
        diagnostic.get("document_type_accuracy")
    )

    if document_type_accuracy is None:
        expected_doc = result["summary_validation"]["document_type_expected"]
        claimed_doc = result["summary_validation"]["document_type_claimed"]

        if (
            expected_doc not in {"unknown", "not_applicable"}
            and claimed_doc not in {"unknown", "not_applicable"}
        ):
            document_type_accuracy = 1.0 if expected_doc == claimed_doc else 0.0

    availability_diag_status = str(
        diagnostic.get("availability_factuality_status")
        or result["availability_factuality"].get("status")
        or "unknown"
    )
    if availability_diag_status not in STATUS_ENUM:
        availability_diag_status = "unknown"

    diagnostic_availability_score = _coerce_float_or_none(
        diagnostic.get("availability_factuality_score")
    )
    if diagnostic_availability_score is None:
        diagnostic_availability_score = availability_score

    diagnostic_trajectory_score = _coerce_float_or_none(
        diagnostic.get("trajectory_consistency_score")
    )
    if diagnostic_trajectory_score is None:
        diagnostic_trajectory_score = trajectory_score

    result["diagnostic_metrics"] = {
        "expected_behavior_accuracy": expected_behavior_accuracy,
        "item_correctness_status": item_status,
        "item_correctness_score": item_score,
        "document_type_accuracy": document_type_accuracy,
        "availability_factuality_status": availability_diag_status,
        "availability_factuality_score": diagnostic_availability_score,
        "trajectory_consistency_score": diagnostic_trajectory_score,
    }

    result["metrics"] = {
        "multi_turn_accuracy_contribution": multi_turn_accuracy_contribution,
        "attempted_turns": attempted_turns,
        "successful_turns": successful_turns,
        "final_outcome_score": 1.0 if fov["final_outcome_success"] else 0.0,
        "expected_behavior_accuracy": expected_behavior_accuracy,
        "item_correctness_score": item_score,
        "document_type_accuracy": document_type_accuracy,
        "availability_factuality_score": diagnostic_availability_score,
        "trajectory_consistency_score": diagnostic_trajectory_score,
    }

    return result


# =========================================================
# 8. Requested item extraction + availability facts
# =========================================================

async def extract_requested_items_with_llm(
    kernel,
    service_id: str,
    *,
    case: dict,
    scenario: Any,
    outcome: dict,
    turns: List[dict],
    catalog_items: List[dict],
    valid_item_nos: List[str],
) -> List[dict]:
    chat_service = kernel.get_service(service_id)

    prompt = f"""
You extract requested catalog items from customer messages for a Sales Order Agent benchmark.

Return only a JSON object matching the schema.
Do not include markdown.
Do not include comments.

Rules:
- Extract requested_items only from customer_text.
- Do not extract requested items from SOA replies.
- Use prior turn context only when the current customer turn depends on references like "it", "same item", "the first one", or "change it to 5".
- Use the Business Central catalog to resolve item_no, description, and UOM when semantically clear.
- If an item cannot be uniquely resolved to the catalog, include description and quantity if available, but item_no should be null.
- Do not evaluate whether the SOA was correct.
- Do not invent item numbers not supported by the catalog.
- Mark requires_item_check=true when the customer asks to create, update, convert, check availability for, add, remove, replace, or modify item lines.
- Mark requires_item_check=false for pure policy, address clarification, or general inquiry turns with no specific item content.

Case:
{json.dumps(case, indent=2, ensure_ascii=False, default=str)}

Scenario:
{json.dumps(scenario, indent=2, ensure_ascii=False, default=str)}

Expected outcome:
{json.dumps(outcome, indent=2, ensure_ascii=False, default=str)}

Business Central catalog items:
{json.dumps(catalog_items, indent=2, ensure_ascii=False, default=str)}

Valid Business Central item numbers:
{json.dumps(valid_item_nos, indent=2, ensure_ascii=False, default=str)}

Turn pairs:
{json.dumps(turns, indent=2, ensure_ascii=False, default=str)}
"""

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a strict JSON-only requested item extraction assistant."
    )
    chat_history.add_user_message(prompt)

    settings = AzureChatPromptExecutionSettings(
        temperature=0.0,
    )

    settings.response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "requested_items_extraction",
            "strict": True,
            "schema": REQUESTED_ITEMS_EXTRACTION_SCHEMA,
        },
    }

    response = await chat_service.get_chat_message_content(
        chat_history,
        settings=settings,
    )

    content = getattr(response, "content", response)
    parsed = _parse_llm_json_or_fail(str(content))

    turn_requested_items = parsed.get("turn_requested_items", [])
    if not isinstance(turn_requested_items, list):
        return []

    normalized = []

    for turn in turn_requested_items:
        if not isinstance(turn, dict):
            continue

        normalized.append({
            "turn_index": int(turn.get("turn_index") or 0),
            "requires_item_check": bool(turn.get("requires_item_check") is True),
            "requested_items": normalize_item_objects(turn.get("requested_items")),
            "reason": str(turn.get("reason") or ""),
        })

    return normalized


def collect_availability_facts_from_extracted_items(
    extracted_turn_items: List[dict],
    client,
) -> List[dict]:
    facts = []
    seen = set()

    for turn in extracted_turn_items or []:
        turn_index = turn.get("turn_index")

        if turn.get("requires_item_check") is not True:
            continue

        for item in turn.get("requested_items", []) or []:
            if not isinstance(item, dict):
                continue

            item_no = item.get("item_no")
            quantity = item.get("quantity")
            uom = item.get("uom") or ""

            if not item_no:
                continue

            try:
                qty_f = float(quantity) if quantity is not None else 1.0
            except Exception:
                qty_f = 1.0

            key = (
                str(item_no).lower(),
                float(qty_f),
                str(uom or "").upper(),
            )

            if key in seen:
                continue

            seen.add(key)

            try:
                api = client.check_availability(
                    item_no=str(item_no),
                    quantity=qty_f,
                    uom_code=uom or "",
                )

                facts.append({
                    "turn_index": turn_index,
                    "item_no": str(item_no).lower(),
                    "description": item.get("description"),
                    "quantity": qty_f,
                    "uom": uom or "",
                    "status": "ok",
                    "api_available": api.get("available") if isinstance(api, dict) else None,
                    "api_raw": api,
                    "source": "bc_check_availability",
                })

            except Exception as e:
                facts.append({
                    "turn_index": turn_index,
                    "item_no": str(item_no).lower(),
                    "description": item.get("description"),
                    "quantity": qty_f,
                    "uom": uom or "",
                    "status": "error",
                    "error": str(e),
                    "source": "bc_check_availability",
                })

    return facts


async def collect_availability_facts_from_extracted_items_async(
    extracted_turn_items: List[dict],
    client,
) -> List[dict]:
    return await asyncio.to_thread(
        collect_availability_facts_from_extracted_items,
        extracted_turn_items,
        client,
    )


# =========================================================
# 9. LLM evaluator
# =========================================================

async def evaluate_with_llm_primary(
    kernel,
    service_id: str,
    *,
    case: dict,
    scenario: Any,
    outcome: dict,
    messages: List[dict],
    optional_expected_items: List[dict],
    extracted_requested_items: List[dict],
    catalog_items: List[dict],
    valid_item_nos: List[str],
    availability_facts: List[dict],
    trajectory_evidence: Dict[str, Any],
    turn_item_evidence: List[dict],
) -> Dict[str, Any]:
    chat_service = kernel.get_service(service_id)

    conversation = build_conversation(messages)
    turns = build_customer_soa_turns(messages)
    final_msg = get_final_soa_message(messages)
    final_soa_text = _text(final_msg) if final_msg else ""

    prompt = f"""
You are the evaluator for Sales Order Agent in Microsoft Dynamics 365 Business Central.

Your job is to evaluate each customer-SOA turn and decide whether the SOA response is successful.

Return only a JSON object matching the provided schema.
Do not include markdown.
Do not include comments.

Primary metric:
- Multi-turn accuracy = successful turns / attempted turns.
- Python computes this from turn_evaluations.
- Your job is to decide turn_success for each turn.

Important evaluation design:
- There is no complete static oracle.
- You cannot inspect real Business Central quote or sales order documents.
- You must not infer hidden quote/order document contents.
- Actual generated items must be extracted only from SOA-visible text.
- Requested items must be extracted from customer-visible text and necessary conversation context.
- Business Central catalog is grounding evidence for item numbers, descriptions, and units of measure.
- Business Central availability_facts are factual evidence from the same availability API the agent can use.
- Optional expected/requested items are weak supporting evidence only and must not override the conversation.

Source hierarchy for item and availability evaluation:
1. Customer request and necessary prior conversation context define what was requested.
2. SOA summary/reply text defines what the agent visibly claims it generated, updated, converted, checked, or answered.
3. Business Central catalog defines valid item numbers, descriptions, and units of measure.
4. Business Central availability_facts define factual availability for requested item_no + quantity + UOM when available.
5. Optional expected/requested items are weak supporting evidence only.

The evaluator cannot read hidden BC documents.
Therefore, item correctness means:
- whether the SOA-visible response/summary is factually consistent with the customer request, BC catalog, and availability_facts,
not whether a hidden quote or sales order document actually contains those lines.

General turn evaluation:
- Judge each turn against the current customer request.
- Use previous context only when the current turn depends on it.
- If the current turn is standalone, evaluate it mainly by itself.
- Do not force every individual turn to satisfy the final outcome by itself.

Context-dependent turns include:
- references such as "it", "that", "this one", "the first one", "the second one"
- confirmations such as "yes", "please proceed", "option B"
- corrections or updates such as changing quantity, replacing item, changing size/color/variant
- answers to earlier clarification questions
- follow-up questions about previous SOA replies

Turn success:
- turn_success=true only when the SOA reply/action is materially correct for the current customer request.
- turn_success=false for material errors:
  - wrong document type
  - ignored current request
  - unnecessary clarification
  - missing required clarification
  - unsupported completion claim
  - unsafe action
  - wrong item facts when item checking is required and visible evidence is sufficient
  - availability contradiction when availability_facts explicitly contradict the SOA claim

Clarification-required turns:
- If expected_turn_behavior="ask_clarification", the primary success criterion is whether the SOA asks the required clarification.
- If the expected outcome or scenario says clarification is required and SOA does not ask the required clarification, mark turn_success=false.
- Prefer failure_reason="missing_clarification" when the required clarification is absent.
- Use "unnecessary_clarification" only when SOA asks for clarification even though the customer request is already sufficiently clear.
- For clarification-required turns, item_check is diagnostic only unless SOA makes an explicit contradictory item or availability claim.
- Do not fail item correctness merely because no quote/order line was created or shown when the correct expected behavior was clarification.

Item check decision:
Set item_check.requires_item_check=true when the turn involves:
- creating quote/order item lines
- updating quote/order item lines
- converting quote to order where item lines should carry over
- checking availability for specific products
- adding/removing/replacing item lines
- changing quantity
- changing explicit size/color/flavor/variant/option
- confirming or selecting a previous item-related proposal

Set item_check.requires_item_check=false when the turn does not involve specific catalog items or document lines.
When false:
- item_check.status="not_applicable"
- item_check.checked_against="not_applicable"
- requested_items=[]
- actual_items=[]
- item correctness must not affect turn_success.

Requested item extraction:
- Prefer the pre-extracted requested items when they are consistent with customer text.
- Extract requested_items from the current customer request and necessary previous context.
- Use previous context only when needed to resolve references like "it", "same item", "the first one", or update instructions.
- Use BC catalog to resolve item_no, description, and UOM when semantically clear.
- If the customer request is ambiguous and no unique catalog item can be resolved, asking clarification can be correct.

Actual item extraction:
- Extract actual_items only from SOA-visible text:
  - the SOA reply/summary for the same turn
  - the final SOA message only if it explicitly lists or summarizes generated quote/order lines
  - previous SOA replies only when the current SOA reply explicitly refers back to previously generated lines
- Do not assume the quote/order contains the correct items just because SOA says it was created.
- Do not infer hidden document lines.
- If SOA text does not expose item lines, actual_items should be [].

Unavailable / not found item handling:
- If the SOA explicitly says a requested item is unavailable, out of stock, not available, or not found, do not classify that item as missing merely because it is not presented as a quoted/created line.
- Treat that as an availability or catalog-matching claim.
- If availability_facts confirm the item is unavailable, do not classify it as missing, wrong, or fabricated.
- Only mark availability_contradiction when SOA's availability claim conflicts with availability_facts.
- If availability_facts are unavailable or errored, mark availability factuality as not_evaluable rather than incorrect.

Missing item rules:
- Use missing_items only when SOA was expected to create, update, or summarize quote/order lines and completely omitted a requested item that should have appeared.
- Do not use missing_items for items that SOA explicitly mentions as unavailable, out of stock, not available, or not found.
- For unavailable or not-found items, record the issue under availability_factuality or reason, not missing_items.

Quantity rules:
- Do not classify quantity as wrong only because SOA-visible text omits the quantity.
- Use wrong_quantity only when SOA-visible text explicitly states a quantity that conflicts with the customer request.
- If quantity is not visible, explain that quantity evidence is insufficient rather than treating it as wrong.
- Missing visible quantities may make item_check.status="not_evaluable" or "partially_correct", but should not populate wrong_quantity unless there is an explicit contradictory quantity.

Item factuality:
- If item checking is required and SOA-visible item facts are present, compare requested_items and actual_items.
- Check:
  - item_no correctness
  - quantity correctness only when SOA explicitly states quantity
  - UOM correctness
  - explicit modifier / variant correctness
  - availability claims against availability_facts when available
- If SOA-visible item facts conflict with the customer request, BC catalog, or availability_facts, item_check.status="incorrect" and turn_success=false when the conflict is material.
- If SOA claims a quote/order was created but does not expose item lines, do not mark item_check.status="incorrect" solely because item lines are absent.
- In that case, use item_check.status="not_evaluable" unless the SOA made a clearly unsupported or contradictory item claim.
- If evidence is insufficient, use item_check.status="not_evaluable" and explain why.
- If SOA fabricates unsupported item numbers or unrelated items, item_check.status="incorrect" and turn_success=false.

For update turns:
- Use context to identify the existing quote/order or item line.
- Evaluate the requested operation: add, remove, replace, change quantity, change variant, or convert.
- If the customer says "make it 5", resolve "it" from context.
- If target is ambiguous, asking clarification can be correct.
- If SOA updates wrong item, wrong quantity, or wrong variant based on visible summary, turn_success=false.

For each item object:
- source_text: exact text span or concise evidence text.
- item_no: item number if present or resolved from BC catalog; otherwise null.
- description: product/item description if present or resolved; otherwise null.
- quantity: requested or actual quantity if present; otherwise null.
- uom: unit of measure if present or resolved; otherwise null.
- modifiers: explicit modifiers such as size, color, flavor, variant, option.
- confidence: high, medium, low, or unknown.

Set item_check.checked_against:
- "customer_context_vs_soa_summary_with_bc_catalog" when comparing requested vs actual items using catalog and/or availability facts.
- "soa_summary_only" only when evaluating internal consistency of SOA summary without enough customer/catalog grounding.
- "not_evaluable" when item checking is required but evidence is insufficient.
- "not_applicable" when item checking is not required.

Failure reason selection rules:
- Use "missing_clarification" when the expected behavior is ask_clarification but the SOA does not ask the required clarification.
- Use "unnecessary_clarification" only when the SOA asks for clarification even though the customer request is already sufficiently clear.
- Use "ignored_current_request" when SOA responds to a different task than the current customer request.
- Use "availability_contradiction" only when availability_facts explicitly contradict SOA's availability claim.
- For outcome_id="clarification_required", if the required clarification is absent, prefer "missing_clarification" over "unnecessary_clarification".

Failure reason guidance:
- wrong_items
- wrong_quantity
- wrong_uom
- wrong_variant_or_modifier
- wrong_document_type
- wrong_workflow_stage
- missing_sales_order_creation
- intervention_not_respected
- unexpected_quote_created
- unnecessary_clarification
- missing_clarification
- ignored_current_request
- unrelated_previous_context_continued
- availability_contradiction
- trajectory_inconsistent
- final_outcome_failed
- missing_soa_reply

Availability factuality:
- Availability is diagnostic unless the current turn is specifically about availability or SOA makes a material availability claim.
- If availability_facts is empty, status="not_evaluable" and score=null.
- Do not penalize only because availability facts are unavailable.
- If SOA says an item is unavailable/out of stock/not available/not found, compare that claim against availability_facts when available.
- If availability_facts confirm the claim, availability_factuality_pass=true.
- If availability_facts contradict the claim, availability_factuality_pass=false and include the contradiction.
- If availability_facts are errored or unavailable, mark not_evaluable, not incorrect.

Summary validation:
- Validate only the final SOA visible text.
- Do not assume the hidden quote/sales order document can be inspected.
- actual_items must be extracted from final_soa_text only if explicitly present.
- If final_soa_text does not include item lines, actual_items should be empty.
- Do not put unavailable/not-found items in missing_items merely because they are not quote/order lines.
- summary_contains_correct_answer=true only if the visible summary provides enough correct information for the requested final answer.

Item correctness scoring:
- 1.0 when all evaluable item-related turns are correct.
- 0.5 when partially correct.
- 0.0 when materially incorrect.
- null when no reliable item evidence.
- status="not_applicable" when no item-related turns.
- status="not_evaluable" when item-related turns exist but SOA-visible evidence is insufficient.

Case:
{json.dumps(case, indent=2, ensure_ascii=False, default=str)}

Scenario:
{json.dumps(scenario, indent=2, ensure_ascii=False, default=str)}

Expected outcome:
{json.dumps(outcome, indent=2, ensure_ascii=False, default=str)}

Optional expected/requested items:
{json.dumps(optional_expected_items, indent=2, ensure_ascii=False, default=str)}

Pre-extracted requested items from customer turns:
{json.dumps(extracted_requested_items, indent=2, ensure_ascii=False, default=str)}

Business Central catalog items:
{json.dumps(catalog_items, indent=2, ensure_ascii=False, default=str)}

Valid Business Central item numbers:
{json.dumps(valid_item_nos, indent=2, ensure_ascii=False, default=str)}

Business Central availability facts:
{json.dumps(availability_facts, indent=2, ensure_ascii=False, default=str)}

Raw SOA trajectory evidence:
{json.dumps(trajectory_evidence, indent=2, ensure_ascii=False, default=str)}

Raw turn item evidence:
{json.dumps(turn_item_evidence, indent=2, ensure_ascii=False, default=str)}

Final SOA message:
{json.dumps(final_soa_text, indent=2, ensure_ascii=False, default=str)}

Turn pairs:
{json.dumps(turns, indent=2, ensure_ascii=False, default=str)}

Full conversation:
{json.dumps(conversation, indent=2, ensure_ascii=False, default=str)}
"""

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a strict JSON-only evaluator for a Sales Order Agent benchmark."
    )
    chat_history.add_user_message(prompt)

    settings = AzureChatPromptExecutionSettings(
        temperature=0.0,
    )

    settings.response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "sales_order_agent_evaluation",
            "strict": True,
            "schema": LLM_EVALUATION_SCHEMA,
        },
    }

    response = await chat_service.get_chat_message_content(
        chat_history,
        settings=settings,
    )

    content = getattr(response, "content", response)
    parsed = _parse_llm_json_or_fail(str(content))
    normalized = normalize_llm_eval_result(parsed)

    return normalized


# =========================================================
# 10. Entrypoint
# =========================================================

async def evaluate_case(
    task_id,
    case: dict,
    scenario: Any,
    outcome: dict,
    messages: List[dict],
    kernel,
    service_id: str,
    client,
) -> Dict[str, Any]:
    catalog_items = await get_catalog_items_async(client)
    valid_item_nos_set = get_valid_item_nos_from_catalog(catalog_items)

    if not valid_item_nos_set:
        valid_item_nos_set = await get_valid_item_nos_async(client)

    valid_item_nos = sorted(list(valid_item_nos_set))

    optional_expected_items = extract_optional_expected_items_from_case(
        case=case,
        outcome=outcome,
        valid_item_nos=valid_item_nos_set,
    )

    scenario_id = None
    outcome_id = None

    if isinstance(case, dict):
        scenario_id = case.get("scenario_id")
        outcome_id = case.get("outcome_id")

    if scenario_id is None and isinstance(scenario, dict):
        scenario_id = scenario.get("scenario_id") or scenario.get("id")

    if outcome_id is None and isinstance(outcome, dict):
        outcome_id = outcome.get("outcome_id") or outcome.get("id")

    trajectory_evidence = analyze_soa_trajectory_for_evidence(
        messages=messages,
    )

    turns = build_customer_soa_turns(messages)

    turn_item_evidence = build_turn_item_evidence(
        turns=turns,
    )

    extracted_requested_items = await extract_requested_items_with_llm(
        kernel=kernel,
        service_id=service_id,
        case=case,
        scenario=scenario,
        outcome=outcome,
        turns=turns,
        catalog_items=catalog_items,
        valid_item_nos=valid_item_nos,
    )

    availability_facts = await collect_availability_facts_from_extracted_items_async(
        extracted_turn_items=extracted_requested_items,
        client=client,
    )

    llm_eval = await evaluate_with_llm_primary(
        kernel=kernel,
        service_id=service_id,
        case=case,
        scenario=scenario,
        outcome=outcome,
        messages=messages,
        optional_expected_items=optional_expected_items,
        extracted_requested_items=extracted_requested_items,
        catalog_items=catalog_items,
        valid_item_nos=valid_item_nos,
        availability_facts=availability_facts,
        trajectory_evidence=trajectory_evidence,
        turn_item_evidence=turn_item_evidence,
    )

    return {
        "task_id": task_id,

        "case_id": case.get("case_id") if isinstance(case, dict) else None,
        "scenario_id": scenario_id,
        "outcome_id": outcome_id,

        "persona_id": case.get("persona_id") if isinstance(case, dict) else None,
        "persona_display_name": case.get("persona_display_name") if isinstance(case, dict) else None,
        "tone_id": case.get("tone_id") if isinstance(case, dict) else None,
        "tone_display_name": case.get("tone_display_name") if isinstance(case, dict) else None,

        "evaluation_mode": "llm_primary_summary_visible_item_extraction",

        "expected": {
            "optional_static_requested_items": optional_expected_items,
            "llm_extracted_requested_items": extracted_requested_items,
        },

        "catalog_items_count": len(catalog_items),
        "valid_item_nos": valid_item_nos,
        "availability_facts": availability_facts,
        "trajectory_analysis": trajectory_evidence,
        "turn_item_evidence": turn_item_evidence,

        "turn_evaluations": llm_eval.get("turn_evaluations", []),
        "turn_metrics": llm_eval.get("turn_metrics", {}),

        "final_outcome_success": bool(llm_eval.get("final_outcome_success")),
        "scenario_success": bool(llm_eval.get("scenario_success")),
        "final_outcome_validation": llm_eval.get("final_outcome_validation", {}),

        "diagnostic_metrics": llm_eval.get("diagnostic_metrics", {}),

        "summary_validation": llm_eval.get("summary_validation", {}),
        "availability_factuality": llm_eval.get("availability_factuality", {}),
        "trajectory_consistency": llm_eval.get("trajectory_consistency", {}),

        "llm_judge": llm_eval,

        "metrics": llm_eval.get("metrics", {}),
        "success": bool(llm_eval.get("success")),
        "failure_reason": llm_eval.get("failure_reason"),
        "reason": llm_eval.get("reason"),
    }


# =========================================================
# 11. Aggregation
# =========================================================

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def get_turn_counts(result: dict) -> Tuple[int, int]:
    tm = result.get("turn_metrics", {}) or {}

    attempted = tm.get("attempted_turns")
    successful = tm.get("successful_turns")

    if isinstance(attempted, int) and isinstance(successful, int):
        return attempted, successful

    turns = result.get("turn_evaluations", []) or []
    if isinstance(turns, list):
        attempted = len(turns)
        successful = sum(
            1 for t in turns
            if isinstance(t, dict) and t.get("turn_success") is True
        )
        return attempted, successful

    return 0, 0


def average_nested_metric(results: List[dict], *path: str) -> Optional[float]:
    values = []

    for r in results:
        current = r
        for p in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(p)

        if isinstance(current, (int, float)):
            values.append(float(current))

    return mean(values) if values else None


def group_by(results: List[dict], key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}

    for r in results:
        value = r.get(key) or "unknown"
        groups.setdefault(str(value), []).append(r)

    return groups


def is_final_outcome_success(result: dict) -> bool:
    fov = result.get("final_outcome_validation", {}) or {}

    if isinstance(fov, dict) and fov.get("final_outcome_success") is True:
        return True

    if result.get("final_outcome_success") is True:
        return True

    return False


def summarize_group(group_results: List[dict]) -> Dict[str, Any]:
    total_attempted_turns = 0
    total_successful_turns = 0
    all_turns_success_count = 0

    for r in group_results:
        attempted, successful = get_turn_counts(r)
        total_attempted_turns += attempted
        total_successful_turns += successful

        if attempted > 0 and attempted == successful:
            all_turns_success_count += 1

    final_success_count = sum(
        1 for r in group_results
        if is_final_outcome_success(r)
    )

    
    expected_behavior_accuracy = average_nested_metric(
        group_results,
        "diagnostic_metrics",
        "expected_behavior_accuracy",
    )


    item_score = average_nested_metric(
        group_results,
        "diagnostic_metrics",
        "item_correctness_score",
    )

    document_type_accuracy = average_nested_metric(
        group_results,
        "diagnostic_metrics",
        "document_type_accuracy",
    )

    availability_score = average_nested_metric(
        group_results,
        "diagnostic_metrics",
        "availability_factuality_score",
    )

    trajectory_score = average_nested_metric(
        group_results,
        "diagnostic_metrics",
        "trajectory_consistency_score",
    )

    return {
        "num_cases": len(group_results),

        "total_attempted_turns": total_attempted_turns,
        "total_successful_turns": total_successful_turns,

        "multi_turn_accuracy": safe_div(
            total_successful_turns,
            total_attempted_turns,
        ),

        "num_all_turns_success_cases": all_turns_success_count,
        "all_turns_success_case_rate": safe_div(
            all_turns_success_count,
            len(group_results),
        ),

        "num_final_outcome_success": final_success_count,
        "final_outcome_accuracy": safe_div(
            final_success_count,
            len(group_results),
        ),
        
        "average_expected_behavior_accuracy": expected_behavior_accuracy,
        "average_item_correctness_score": item_score,
        "average_document_type_accuracy": document_type_accuracy,
        "average_availability_factuality_score": availability_score,
        "average_trajectory_consistency_score": trajectory_score,
    }


def aggregate_eval_results(results: List[dict]) -> Dict[str, Any]:
    if not results:
        return {
            "primary_metric": "multi_turn_accuracy",
            "num_cases": 0,
            "num_evaluated_cases": 0,
            "num_error": 0,
            "multi_turn_accuracy": 0.0,
            "final_outcome_accuracy": 0.0,
            "overall": {},
            "by_persona": {},
            "by_tone": {},
            "by_scenario": {},
            "error_case_ids": [],
            "failure_reason_counts": {},
            "failures": [],
        }

    error_cases = [r for r in results if r.get("status") == "error"]
    normal_cases = [r for r in results if r.get("status") != "error"]

    overall = summarize_group(normal_cases)

    by_persona = {
        persona: summarize_group(group)
        for persona, group in group_by(normal_cases, "persona_id").items()
    }

    by_tone = {
        tone: summarize_group(group)
        for tone, group in group_by(normal_cases, "tone_id").items()
    }

    by_scenario = {
        scenario: summarize_group(group)
        for scenario, group in group_by(normal_cases, "scenario_id").items()
    }

    failure_reason_counts: Dict[str, int] = {}
    failures = []

    for r in normal_cases:
        attempted, successful = get_turn_counts(r)
        final_success = is_final_outcome_success(r)
        turn_accuracy_contribution = safe_div(successful, attempted)
        all_turns_success = attempted > 0 and attempted == successful

        if all_turns_success:
            continue

        reason_key = r.get("failure_reason") or "unknown"
        failure_reason_counts[reason_key] = failure_reason_counts.get(reason_key, 0) + 1

        summary_validation = r.get("summary_validation", {}) or {}
        availability_factuality = r.get("availability_factuality", {}) or {}
        trajectory_consistency = r.get("trajectory_consistency", {}) or {}
        diagnostic_metrics = r.get("diagnostic_metrics", {}) or {}
        final_outcome_validation = r.get("final_outcome_validation", {}) or {}

        failures.append({
            "case_id": r.get("case_id"),
            "scenario_id": r.get("scenario_id"),
            "outcome_id": r.get("outcome_id"),
            "persona_id": r.get("persona_id"),
            "tone_id": r.get("tone_id"),
            "evaluation_mode": r.get("evaluation_mode"),

            "success": r.get("success"),
            "all_turns_success": all_turns_success,
            "final_outcome_success": final_success,
            "failure_reason": r.get("failure_reason"),
            "reason": r.get("reason"),

            "attempted_turns": attempted,
            "successful_turns": successful,
            "multi_turn_accuracy_contribution": turn_accuracy_contribution,

            "expected_outcome": final_outcome_validation.get("expected_outcome"),
            "actual_outcome": final_outcome_validation.get("actual_outcome"),
            "final_outcome_reason": final_outcome_validation.get("reason"),

            "item_correctness_status": diagnostic_metrics.get("item_correctness_status"),
            "item_correctness_score": diagnostic_metrics.get("item_correctness_score"),
            "document_type_accuracy": diagnostic_metrics.get("document_type_accuracy"),

            "summary_status": summary_validation.get("status"),
            "summary_missing_items": summary_validation.get("missing_items"),
            "summary_wrong_quantity": summary_validation.get("wrong_quantity"),
            "summary_extra_items": summary_validation.get("extra_items"),

            "policy_refusal_detected": summary_validation.get("policy_refusal_detected"),
            "sensitive_data_leak_detected": summary_validation.get("sensitive_data_leak_detected"),

            "availability_status": availability_factuality.get("status"),
            "availability_factuality_score": availability_factuality.get("availability_factuality_score"),
            "availability_inconsistent_claims": availability_factuality.get("inconsistent_claims"),

            "trajectory_consistent": trajectory_consistency.get("trajectory_consistent"),
            "trajectory_consistency_score": trajectory_consistency.get("trajectory_consistency_score"),
            "quantity_contradictions": trajectory_consistency.get("quantity_contradictions"),
            "other_contradictions": trajectory_consistency.get("other_contradictions"),

            "turn_evaluations": r.get("turn_evaluations", []),
        })

    return {
        "primary_metric": "multi_turn_accuracy",

        "num_cases": len(results),
        "num_evaluated_cases": len(normal_cases),
        "num_error": len(error_cases),
        "error_case_ids": [r.get("case_id") for r in error_cases],

        "overall": overall,
        "by_persona": by_persona,
        "by_tone": by_tone,
        "by_scenario": by_scenario,

        "multi_turn_accuracy": overall.get("multi_turn_accuracy", 0.0),
        "pass_rate": overall.get("multi_turn_accuracy", 0.0),

        "all_turns_success_case_rate": overall.get("all_turns_success_case_rate"),
        "final_outcome_accuracy": overall.get("final_outcome_accuracy", 0.0),

        "average_expected_behavior_accuracy": overall.get("average_expected_behavior_accuracy"),
        "average_item_correctness_score": overall.get("average_item_correctness_score"),
        "average_document_type_accuracy": overall.get("average_document_type_accuracy"),
        "average_availability_factuality_score": overall.get("average_availability_factuality_score"),
        "average_trajectory_consistency_score": overall.get("average_trajectory_consistency_score"),

        "failure_reason_counts": failure_reason_counts,
        "failures": failures,
    }
