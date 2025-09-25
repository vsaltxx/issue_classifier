#!/usr/bin/env python3
import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG (edit as needed)
# =========================
TRAIN_PATH       = "v_data/done_representatives.jsonl"
TEST_PATH        = "v_data/test_done_issues.jsonl"
COMPONENTS_PATH  = "v_data/components.json"
OUT_PREDS_PATH   = "predictions.json"

MODEL            = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DESC_MAX         = 300
TRAIN_LIMIT      = 700
TEST_LIMIT       = 20
BATCH_SIZE       = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assign-components")

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# =========================
# Data model
# =========================
@dataclass
class IssueRow:
    issue_key: str
    summary: str
    description: str
    components: List[str]  # may be empty for TEST items

# =========================
# IO helpers
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def read_components(path: str) -> List[str]:
    """
    Read components from a JSON array file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def normalize_issue(raw: Dict[str, Any], desc_max: int) -> IssueRow:
    key = (raw.get("issue_key") or raw.get("key") or "").strip()
    summary = (raw.get("summary") or "").strip()
    desc = (raw.get("description") or "")[:desc_max]
    comps = raw.get("components") or []
    comps = [str(c).strip() for c in comps if str(c).strip()]
    return IssueRow(issue_key=key, summary=summary, description=desc, components=comps)

# =========================
# Prompting
# =========================
SYSTEM_MSG = (
    "You are a classifier that assigns exactly one ESP-IDF Jira component to each TEST issue. "
    "Use only values from Allowed components exactly as written (case-sensitive). "
    "Return only valid JSON that satisfies the provided schema."
)

def build_payload(train_rows: List[IssueRow], test_rows: List[IssueRow]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = [
        {
            "issue_key": r.issue_key,
            "summary": r.summary,
            "description": r.description,
            "components": r.components,
        }
        for r in train_rows
    ]
    test = [
        {
            "issue_key": r.issue_key,
            "summary": r.summary,
            "description": r.description,
            "components": [],  # must be predicted
        }
        for r in test_rows
    ]
    return train, test

def make_user_prompt(train_payload: List[Dict[str, Any]], test_payload: List[Dict[str, Any]], allowed_components: List[str]) -> str:
    return (
        "Assign exactly one component to each TEST issue.\n\n"
        "Rules:\n"
        "- Choose exactly one best component per TEST issue.\n"
        "- Use ONLY values from Allowed components (case-sensitive).\n"
        "- Output VALID JSON only (per schema).\n\n"
        f"Allowed components: {allowed_components}\n\n"
        "TRAIN examples (JSON array):\n"
        f"{json.dumps(train_payload, ensure_ascii=False)}\n\n"
        "TEST issues (JSON array):\n"
        f"{json.dumps(test_payload, ensure_ascii=False)}"
    )

def response_schema() -> Dict[str, Any]:
    # Responses API requires a root object schema; wrap array in "predictions"
    return {
        "type": "json_schema",
        "name": "assign_components",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_key": {"type": "string"},
                            "components": {"type": "string"}
                        },
                        "required": ["issue_key", "components"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["predictions"],
            "additionalProperties": False
        },
        "strict": True
    }

# =========================
# OpenAI call
# =========================
def predict_batch(
    client: OpenAI,
    model: str,
    train_rows: List[IssueRow],
    test_rows: List[IssueRow],
    allowed_components: List[str],
) -> List[Dict[str, str]]:
    train_payload, test_payload = build_payload(train_rows, test_rows)
    prompt = make_user_prompt(train_payload, test_payload, allowed_components)

    resp = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        text={"format": response_schema()},
        timeout=120,
    )
    obj = json.loads(resp.output_text)  # guaranteed JSON per schema
    out = obj.get("predictions", [])
    allowed = set(allowed_components)

    cleaned: List[Dict[str, str]] = []
    for item in out:
        key = str(item.get("issue_key", "")).strip()
        comp = str(item.get("components", "")).strip()
        if comp not in allowed:
            # fallback: force a valid label to avoid eval crashes; or skip
            comp = next(iter(allowed), "")
        if key and comp:
            cleaned.append({"issue_key": key, "components": comp})
    return cleaned

# =========================
# Evaluation
# =========================
def evaluate_accuracy(gold_rows: List[IssueRow], preds: List[Dict[str, str]]) -> Dict[str, Any]:
    gold_map: Dict[str, List[str]] = {r.issue_key: r.components for r in gold_rows if r.issue_key and r.components}
    pred_map: Dict[str, str] = {p["issue_key"]: p["components"] for p in preds if p.get("issue_key") and p.get("components")}
    total = correct = 0
    for key, gold_labels in gold_map.items():
        pred = pred_map.get(key)
        if pred is None:
            continue
        total += 1
        if pred in gold_labels:
            correct += 1
        else:
            logging.info("Incorrect: %s → %s not in %s", key, pred, gold_labels)
    acc = (correct / total) if total else 0.0
    return {"evaluated": total, "correct": correct, "accuracy": acc}

# =========================
# Utils
# =========================
def chunk(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

# =========================
# Main
# =========================
def main() -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)

    log.info("Loading data …")
    raw_train = read_jsonl(TRAIN_PATH)
    raw_test  = read_jsonl(TEST_PATH)
    comps     = read_components(COMPONENTS_PATH)
    log.info("Loaded train=%d test=%d components=%d", len(raw_train), len(raw_test), len(comps))

    train_rows = [normalize_issue(r, DESC_MAX) for r in raw_train[: TRAIN_LIMIT]]
    test_rows_all = [normalize_issue(r, DESC_MAX) for r in raw_test[: TEST_LIMIT]]

    all_preds: List[Dict[str, str]] = []
    batches = chunk(test_rows_all, BATCH_SIZE) if test_rows_all else []
    log.info("Predicting in %d batch(es), batch_size=%d", len(batches), BATCH_SIZE)

    for i, batch in enumerate(batches, start=1):
        log.info("Batch %d/%d (issues=%d)", i, len(batches), len(batch))
        preds = predict_batch(client, MODEL, train_rows, batch, comps)
        all_preds.extend(preds)

    with open(OUT_PREDS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)
    log.info("Saved predictions → %s (%d items)", OUT_PREDS_PATH, len(all_preds))

    metrics = evaluate_accuracy(test_rows_all, all_preds)
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()
