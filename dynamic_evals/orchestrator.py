# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import asyncio
import json
import os
import traceback
import time
from datetime import datetime, timezone
import uuid
import hashlib

from agent_task_client import AgentTaskClient
from reviewer_agent import review_one_attention_step
from customer_agent import (
    generate_runtime_followup,
    build_kernel,
    load_personas,
    load_tones,
    build_persona_index,
    build_tone_index,
    find_scenario_and_outcome,
    build_thread_context,
)
from evaluator import evaluate_case, aggregate_eval_results

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from azure.identity import DefaultAzureCredential


AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_DEPLOYMENT = "gpt-5.4-mini"
AZURE_OPENAI_API_VERSION = ""
SERVICE_ID = "evaluator-llm"

TASK_OUTPUTS_DIR = "task_outputs"
EVAL_RESULTS_DIR = "eval_results"
LOGS_DIR = "logs"
CHECKPOINT_FILE = os.path.join(EVAL_RESULTS_DIR, "_checkpoint.json")
SUMMARY_FILE = os.path.join(EVAL_RESULTS_DIR, "summary.json")
SUMMARY_LATEST_FILE = os.path.join(EVAL_RESULTS_DIR, "summary_latest.json")


# =========================================================
# Helpers
# =========================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_case_filename(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(case_id))


def eval_result_path(case_id: str) -> str:
    return os.path.join(EVAL_RESULTS_DIR, f"{safe_case_filename(case_id)}.json")


def task_output_path(case_id: str) -> str:
    return os.path.join(TASK_OUTPUTS_DIR, f"{safe_case_filename(case_id)}.json")


def log_path(case_id: str) -> str:
    return os.path.join(LOGS_DIR, f"{safe_case_filename(case_id)}.json")


def error_log_path(case_id: str) -> str:
    return os.path.join(LOGS_DIR, f"{safe_case_filename(case_id)}.error.txt")


def atomic_write_json(path: str, data, *, retries: int = 10, delay_seconds: float = 0.25) -> bool:
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)

    path_hash = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    short_uuid = uuid.uuid4().hex[:8]

    tmp_path = os.path.join(
        target_dir,
        f".tmp_{path_hash}_{os.getpid()}_{short_uuid}.json"
    )

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    last_error = None

    for attempt in range(retries):
        try:
            os.replace(tmp_path, path)
            return True
        except PermissionError as e:
            last_error = e
            time.sleep(delay_seconds * (attempt + 1))
        except OSError as e:
            last_error = e
            time.sleep(delay_seconds * (attempt + 1))

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    raise last_error


def safe_write_checkpoint_json(path: str, data) -> bool:
    try:
        return atomic_write_json(path, data, retries=12, delay_seconds=0.25)
    except Exception as e:
        print(f"[warn] failed to write checkpoint file {path}: {type(e).__name__}: {e}")
        return False


def load_json_if_exists(path: str):
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_finished_eval_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return False

    if result.get("status") == "running":
        return False

    if result.get("status") == "error":
        return True

    if "turn_metrics" in result or "success" in result:
        return True

    return False


def is_finished_task_output(record: dict) -> bool:
    if not isinstance(record, dict):
        return False

    if record.get("status") == "running":
        return False

    if record.get("status") == "error":
        return True

    if record.get("messages") is not None and record.get("task_id") is not None:
        return True

    return False


def load_existing_eval_results() -> list:
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    results = []

    for name in sorted(os.listdir(EVAL_RESULTS_DIR)):
        if not name.endswith(".json"):
            continue

        if name in {"_checkpoint.json", "summary.json", "summary_latest.json"}:
            continue

        data = load_json_if_exists(os.path.join(EVAL_RESULTS_DIR, name))

        if is_finished_eval_result(data):
            results.append(data)

    return results


def build_eval_kernel() -> Kernel:
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


def make_client(args) -> AgentTaskClient:
    return AgentTaskClient(
        base_url=args.base_url,
        company=args.company,
        user=args.user,
        password=args.password,
    )


def write_summary_checkpoint(
    *,
    all_eval_results: list,
    total_cases: int,
    processed_cases: int,
    skipped_cases: int,
    current_case_id: str | None = None,
):
    summary = aggregate_eval_results(all_eval_results)

    checkpoint = {
        "updated_at_utc": utc_now_iso(),
        "total_cases_discovered": total_cases,
        "processed_cases_this_run": processed_cases,
        "skipped_cases_this_run": skipped_cases,
        "num_results_on_disk_or_memory": len(all_eval_results),
        "current_case_id": current_case_id,
        "summary": summary,
    }

    safe_write_checkpoint_json(SUMMARY_LATEST_FILE, summary)
    safe_write_checkpoint_json(CHECKPOINT_FILE, checkpoint)

    print("\n========== CHECKPOINT ==========")
    print(json.dumps(checkpoint, indent=2, ensure_ascii=False))


# =========================================================
# Orchestrator
# =========================================================

class Orchestrator:
    def __init__(self, client, exec_kernel):
        self.client = client
        self.exec_kernel = exec_kernel

    async def run_case_execution(
        self,
        case,
        scenario,
        outcome,
        persona,
        tone,
        follow_up_plan,
        *,
        source_path: str,
        max_steps=30,
        max_followups=10,
        poll_seconds=1.0,
    ):
        case_id = case["case_id"]

        initial = case["initial_turn"]
        sender = initial.get("from") or initial.get("from_")
        message = initial.get("body") or initial.get("message") or ""
        original_customer_request = message

        safe_external_id = f"case-{uuid.uuid4()}"

        result = self.client.create_task_and_wait(
            from_text=sender,
            message_text=message,
            external_id=safe_external_id,
        )

        task_id = result["taskId"]
        print(f"[orchestrator] case_id={case_id}")
        print(f"[orchestrator] task_id={task_id}")
        print(f"[orchestrator] external_id={safe_external_id}")

        last_soa_message_id = None
        followup_count = 0
        final_state = None

        try:
            for step in range(max_steps):
                task = self.client.get_task(task_id)
                status = (task.get("status") or "").lower()
                needs_attention = task.get("needsAttention")

                print(f"[case={case_id}] [step={step}] status={status} needsAttention={needs_attention}")

                if status in {"completed", "failed", "canceled", "rejected"}:
                    final_state = {
                        "state": "terminal",
                        "status": status,
                        "task": task,
                    }
                    break

                if needs_attention:
                    decision = await review_one_attention_step(self.client, task_id)

                    if decision["action"] == "stop":
                        final_state = {
                            "state": "stopped_by_reviewer",
                            "status": status,
                            "task": task,
                            "decision": decision,
                        }
                        break

                    continue

                if status == "paused":
                    messages = self.client.get_messages(task_id)
                    soa_msg = self._latest_soa_output_message(messages)

                    if not soa_msg:
                        final_state = {
                            "state": "no_soa_reply",
                            "status": status,
                            "task": task,
                        }
                        break

                    soa_reply = self._message_text(soa_msg)
                    soa_msg_id = soa_msg.get("id") or soa_msg.get("systemId") or hash(soa_reply)

                    if not soa_reply:
                        final_state = {
                            "state": "empty_soa_reply",
                            "status": status,
                            "task": task,
                        }
                        break

                    if soa_msg_id == last_soa_message_id:
                        await asyncio.sleep(poll_seconds)
                        continue

                    last_soa_message_id = soa_msg_id

                    if not self._should_follow_up(
                        soa_reply,
                        follow_up_plan,
                        followup_count,
                        max_followups,
                    ):
                        final_state = {
                            "state": "waiting_for_customer_no_more_followups",
                            "status": status,
                            "task": task,
                        }
                        break

                    selected_sender = {
                        "name": case["sender_name"],
                        "email": case["sender_email"],
                        "sender_type": case["sender_type"],
                    }

                    if follow_up_plan.get("follow_up_sender_email"):
                        selected_sender = {
                            "name": follow_up_plan.get("follow_up_sender_name", case["sender_name"]),
                            "email": follow_up_plan.get("follow_up_sender_email", case["sender_email"]),
                            "sender_type": follow_up_plan.get("follow_up_sender_type", case["sender_type"]),
                        }

                    followup = await generate_runtime_followup(
                        kernel=self.exec_kernel,
                        scenario=scenario,
                        outcome=outcome,
                        persona=persona,
                        tone=tone,
                        selected_sender=selected_sender,
                        follow_up_plan=follow_up_plan,
                        previous_soa_reply=soa_reply,
                        thread_context=build_thread_context(messages),
                        original_customer_request=original_customer_request,
                    )

                    print(f"[case={case_id}] sending customer follow-up:")
                    print(followup.body)

                    self.client.add_message(
                        task_id,
                        from_text=followup.from_,
                        message_text=followup.body,
                    )

                    followup_count += 1
                    continue

                await asyncio.sleep(poll_seconds)

            if final_state is None:
                task = self.client.get_task(task_id)
                final_state = {
                    "state": "max_steps_reached",
                    "status": task.get("status"),
                    "task": task,
                }

            messages = self.client.get_messages(task_id)
            task = self.client.get_task(task_id)

            output_record = {
                "case_id": case_id,
                "task_id": task_id,
                "external_id": safe_external_id,
                "source_path": source_path,
                "scenario_id": scenario.get("scenario_id") if isinstance(scenario, dict) else None,
                "outcome_id": outcome.get("outcome_id") if isinstance(outcome, dict) else None,
                "persona_id": case.get("persona_id"),
                "tone_id": case.get("tone_id"),
                "input_case": case,
                "task": task,
                "messages": messages,
                "final_state": final_state,
                "completed_at_utc": utc_now_iso(),
            }

            os.makedirs(TASK_OUTPUTS_DIR, exist_ok=True)
            atomic_write_json(task_output_path(case_id), output_record)

            os.makedirs(LOGS_DIR, exist_ok=True)
            atomic_write_json(log_path(case_id), messages)

            return output_record

        except Exception:
            messages = []
            task = None

            try:
                messages = self.client.get_messages(task_id)
            except Exception:
                pass

            try:
                task = self.client.get_task(task_id)
            except Exception:
                pass

            error_record = {
                "case_id": case_id,
                "task_id": task_id,
                "external_id": safe_external_id,
                "source_path": source_path,
                "status": "error",
                "error_type": "execution_error",
                "traceback": traceback.format_exc(),
                "input_case": case,
                "task": task,
                "messages": messages,
                "failed_at_utc": utc_now_iso(),
            }

            atomic_write_json(task_output_path(case_id), error_record)
            raise

    def _message_text(self, msg: dict) -> str:
        return str(
            msg.get("body")
            or msg.get("messageContent")
            or msg.get("content")
            or ""
        ).strip()

    def _is_soa_output_message(self, msg: dict) -> bool:
        direction = str(msg.get("direction") or "").lower()
        msg_type = str(msg.get("type") or "").lower()

        author = " ".join([
            str(msg.get("from") or ""),
            str(msg.get("fromText") or ""),
            str(msg.get("author") or ""),
            str(msg.get("sender") or ""),
            str(msg.get("role") or ""),
        ]).lower()

        return (
            direction == "output"
            or msg_type in {"output", "assistant", "reply"}
            or "sales order agent" in author
            or "soa" in author
            or "assistant" in author
        )

    def _latest_soa_output_message(self, messages: list) -> dict | None:
        for msg in reversed(messages or []):
            text = self._message_text(msg)
            if not text:
                continue

            if self._is_soa_output_message(msg):
                return msg

        return None

    def _looks_like_clarification_request(self, text: str) -> bool:
        text = (text or "").lower()

        clarification_signals = [
            "please review",
            "let us know",
            "which you would like",
            "which one",
            "please specify",
            "specify your selection",
            "kindly specify",
            "please confirm",
            "confirm which",
            "choose",
            "selection",
            "options",
            "available options",
            "if you require further details",
            "if you have a preference",
            "what would you like",
            "which item",
            "which option",
        ]

        return any(s in text for s in clarification_signals)

    def _should_follow_up(
        self,
        soa_reply: str,
        follow_up_plan: dict,
        followup_count: int,
        max_followups: int,
    ) -> bool:
        if follow_up_plan.get("mode") != "runtime_grounded":
            return False

        plan_limit = int(follow_up_plan.get("max_additional_turns", max_followups) or 0)
        effective_limit = min(max_followups, plan_limit)

        if followup_count >= effective_limit:
            return False

        text = (soa_reply or "").strip().lower()

        if len(text) < 10:
            return False

        hard_stop_signals = [
            "order confirmed",
            "sales order has been created",
            "your order is complete",
            "no further action is required",
        ]

        if any(k in text for k in hard_stop_signals):
            return False

        if self._looks_like_clarification_request(text):
            return True

        return True


# =========================================================
# Discovery / execution / evaluation
# =========================================================

def discover_cases(tasks_input_dir: str) -> list:
    run_items = []

    for root, dirs, files in os.walk(tasks_input_dir):
        dirs.sort()

        for f in sorted(files):
            if not f.endswith(".json"):
                continue

            path = os.path.join(root, f)

            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            domain = os.path.basename(os.path.dirname(path))

            scenario, outcome = find_scenario_and_outcome(
                domain,
                data["scenario_id"],
                data["outcome_id"],
            )

            for case in data.get("test_cases", []):
                run_items.append({
                    "path": path,
                    "domain": domain,
                    "case": case,
                    "scenario": scenario,
                    "outcome": outcome,
                    "scenario_id": data["scenario_id"],
                    "outcome_id": data["outcome_id"],
                })

    return run_items


def print_execute_resume_preflight(run_items):
    finished = 0
    unfinished = []
    unreadable = 0
    missing = 0

    for idx, item in enumerate(run_items, start=1):
        case = item["case"]
        case_id = case.get("case_id", "UNKNOWN_CASE")
        output_path = task_output_path(case_id)

        exists = os.path.exists(output_path)
        existing_output = load_json_if_exists(output_path)

        if exists and existing_output is None:
            unreadable += 1

        if not exists:
            missing += 1

        if is_finished_task_output(existing_output):
            finished += 1
        else:
            unfinished.append({
                "index": idx,
                "case_id": case_id,
                "output_path": output_path,
                "exists": exists,
                "status": existing_output.get("status") if isinstance(existing_output, dict) else None,
                "has_task_id": isinstance(existing_output, dict) and existing_output.get("task_id") is not None,
                "has_messages": isinstance(existing_output, dict) and existing_output.get("messages") is not None,
                "messages_count": len(existing_output.get("messages", [])) if isinstance(existing_output, dict) and isinstance(existing_output.get("messages"), list) else None,
                "completed_at_utc": existing_output.get("completed_at_utc") if isinstance(existing_output, dict) else None,
                "source_path": item.get("path"),
            })

    print("\n========== EXECUTE RESUME PREFLIGHT ==========")
    print(f"total discovered cases: {len(run_items)}")
    print(f"finished outputs recognized: {finished}/{len(run_items)}")
    print(f"unfinished outputs recognized: {len(unfinished)}/{len(run_items)}")
    print(f"missing output files: {missing}")
    print(f"unreadable/bad json output files: {unreadable}")

    print("\nFirst 20 unfinished cases:")
    for x in unfinished[:20]:
        print(json.dumps(x, indent=2, ensure_ascii=False))

    print("========== END PREFLIGHT ==========\n")

async def execute_one_item(item, args, persona_index, tone_index, semaphore):
    case = item["case"]
    case_id = case.get("case_id", "UNKNOWN_CASE")
    output_path = task_output_path(case_id)

    async with semaphore:
        existing_output = load_json_if_exists(output_path)

        if args.resume and not args.force_rerun and is_finished_task_output(existing_output):
            print(f"[execute resume] skip finished output: {case_id}")
            return existing_output

        running_marker = {
            "case_id": case_id,
            "status": "running",
            "started_at_utc": utc_now_iso(),
            "source_path": item["path"],
        }
        atomic_write_json(output_path, running_marker)

        client = make_client(args)
        exec_kernel = build_kernel()
        orchestrator = Orchestrator(client, exec_kernel)

        persona = persona_index[case["persona_id"]]
        tone = tone_index[case["tone_id"]]

        try:
            return await orchestrator.run_case_execution(
                case=case,
                scenario=item["scenario"],
                outcome=item["outcome"],
                persona=persona,
                tone=tone,
                follow_up_plan=case.get("follow_up_plan", {}),
                source_path=item["path"],
            )

        except Exception as e:
            print(f"❌ EXECUTION FAILED: {case_id}")
            print(traceback.format_exc())

            error_record = {
                "case_id": case_id,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "failed_at_utc": utc_now_iso(),
                "source_path": item["path"],
            }

            atomic_write_json(output_path, error_record)

            with open(error_log_path(case_id), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

            return error_record


async def execute_all(run_items, args, persona_index, tone_index):
    os.makedirs(TASK_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        execute_one_item(item, args, persona_index, tone_index, semaphore)
        for item in run_items
    ]

    results = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1

        print(f"[execute progress] {completed}/{len(run_items)}")

    return results


async def evaluate_output_record(record, args, eval_kernel):
    case_id = record.get("case_id")
    result_path = eval_result_path(case_id)

    if args.resume and not args.force_rerun:
        existing_result = load_json_if_exists(result_path)
        if is_finished_eval_result(existing_result):
            print(f"[evaluate resume] skip finished eval: {case_id}")
            return existing_result

    if record.get("status") == "error":
        error_result = {
            "case_id": case_id,
            "status": "error",
            "success": False,
            "error_type": record.get("error_type", "execution_error"),
            "error_message": record.get("error_message"),
            "traceback": record.get("traceback"),
            "source_path": record.get("source_path"),
            "failed_at_utc": utc_now_iso(),
        }
        atomic_write_json(result_path, error_result)
        return error_result

    case = record["input_case"]
    messages = record.get("messages", [])

    domain = os.path.basename(os.path.dirname(record.get("source_path", "")))
    scenario_id = record.get("scenario_id")
    outcome_id = record.get("outcome_id")

    scenario, outcome = find_scenario_and_outcome(domain, scenario_id, outcome_id)

    client = make_client(args)

    running_marker = {
        "case_id": case_id,
        "status": "running",
        "started_at_utc": utc_now_iso(),
        "source_path": record.get("source_path"),
    }
    atomic_write_json(result_path, running_marker)

    eval_result = await evaluate_case(
        task_id=record.get("task_id"),
        case=case,
        scenario=scenario,
        outcome=outcome,
        messages=messages,
        kernel=eval_kernel,
        service_id=SERVICE_ID,
        client=client,
    )

    eval_result.pop("status", None)
    eval_result["completed_at_utc"] = utc_now_iso()
    eval_result["source_path"] = record.get("source_path")
    eval_result["task_output_path"] = task_output_path(case_id)

    atomic_write_json(result_path, eval_result)

    return eval_result


async def evaluate_all(args):
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    eval_kernel = build_eval_kernel()

    records = []

    if not os.path.exists(TASK_OUTPUTS_DIR):
        print(f"No {TASK_OUTPUTS_DIR} directory found.")
        return []

    for name in sorted(os.listdir(TASK_OUTPUTS_DIR)):
        if not name.endswith(".json"):
            continue

        record = load_json_if_exists(os.path.join(TASK_OUTPUTS_DIR, name))
        if is_finished_task_output(record):
            records.append(record)

    print(f"[evaluate] found {len(records)} task output records")

    results = []
    completed = 0

    for record in records:
        result = await evaluate_output_record(record, args, eval_kernel)
        results.append(result)
        completed += 1

        if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
            summary = aggregate_eval_results(load_existing_eval_results())
            safe_write_checkpoint_json(SUMMARY_LATEST_FILE, summary)
            print(f"[evaluate checkpoint] {completed}/{len(records)}")

    final_results = load_existing_eval_results()
    summary = aggregate_eval_results(final_results)

    safe_write_checkpoint_json(SUMMARY_FILE, summary)
    safe_write_checkpoint_json(SUMMARY_LATEST_FILE, summary)

    checkpoint = {
        "updated_at_utc": utc_now_iso(),
        "status": "finished",
        "num_final_results": len(final_results),
        "summary": summary,
    }

    safe_write_checkpoint_json(CHECKPOINT_FILE, checkpoint)

    print("\n========== EVAL SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return results


# =========================================================
# Main
# =========================================================

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["execute", "evaluate", "all"], default="all")

    parser.add_argument("--base-url", required=True)
    parser.add_argument("--company", default="")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--tasks-input-dir", default="tasks_input")

    parser.add_argument("--concurrency", type=int, default=5)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(TASK_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    personas = load_personas("datasets/personas.yaml")
    tones = load_tones("datasets/tones.yaml")

    persona_index = build_persona_index(personas)
    tone_index = build_tone_index(tones)

    run_items = discover_cases(args.tasks_input_dir)

    print(f"\n========== DISCOVERED {len(run_items)} CASES ==========")
    print(f"mode={args.mode}")
    print(f"concurrency={args.concurrency}")

    if args.mode in {"execute", "all"}:     
        if args.resume and not args.force_rerun:
            print_execute_resume_preflight(run_items)
        await execute_all(run_items, args, persona_index, tone_index)

    if args.mode in {"evaluate", "all"}:
        await evaluate_all(args)


if __name__ == "__main__":
    asyncio.run(main())