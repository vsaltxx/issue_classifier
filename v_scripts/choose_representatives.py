import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

NUM_REPRESENTATIVES = 700


def load_data(file_path, limit: int | None = None):
    with open(file_path, "r") as f:
        if limit is None:
            return [json.loads(line) for line in f]
        rows = []
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            rows.append(json.loads(line))
        return rows

def load_allowed_components():
    with open("components.json", "r") as f:
        return json.load(f)

def format_prompt(train_data, allowed_components):
    prompt = f"""
    You are selecting representative ESP-IDF Jira issues from TRAIN.

    Objective
    - Select exactly {NUM_REPRESENTATIVES} issues that best cover the variety of components and topics.

    Rules
    - Prefer clear, canonical examples with strong signal between summary/description and the labeled component.
    - Ensure diversity across Allowed components; avoid near-duplicates.
    - Output MUST be CSV only as specified below. No prose, no markdown fences, no quotes.

    Input
    - TRAIN: JSON array of objects with fields: issue_key, summary, description, components (contains the correct component for the issue).

    Output
    - Return ONLY a single line CSV string with exactly {NUM_REPRESENTATIVES} values.
    - Each value is the issue_key of a selected TRAIN issue, exactly as it appears in TRAIN (e.g., IDFGH-123).
    - Format: issue_key values separated by commas (no spaces), no header, no trailing comma.

    TRAIN:
    {train_data}

    """
    return prompt


def main():
    train_data = load_data("v_data/train_done_issues.jsonl", limit=1500)

    def _truncate_desc(row):
        desc = row.get("description", "")
        row["description"] = (desc or "")[:100]
        return row

    train_data = [_truncate_desc(dict(d)) for d in train_data]

    allowed_components = load_allowed_components()
    prompt = format_prompt(train_data, allowed_components)
    # call the model
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You output only CSV as instructed."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    with open("v_data/representatives.txt", "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()