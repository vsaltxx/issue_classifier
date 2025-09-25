import json
import os
from jira import JIRA
from jira.exceptions import JIRAError
from jira.resources import Issue as JiraIssue 
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

PROJECT_KEY = "IDFGH"

DEFAULT_FIELDS = "summary,description,components"
def connect_jira() -> JIRA:
    jira_url = "https://jira.espressif.com:8443/"
    jira_token = os.environ["JIRA_API_TOKEN"]
    return JIRA(server=jira_url, token_auth=jira_token)


def issue_to_row(issue: JiraIssue) -> Dict[str, Any]:
    f = issue.fields
    comps = getattr(f, "components", None) or []  # ensure list
    return {
        "key": issue.key,
        "summary": f.summary or "",
        "description": (getattr(f, "description", "") or "")[:300],
        "components": [c.name for c in comps],
    }

def fetch_done_issues(jira: JIRA, jql: str, fields: str = DEFAULT_FIELDS,
                      require_components: bool = False, page_size: int = 100) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start_at = 0
    while True:
        batch = jira.search_issues(
            jql, startAt=start_at, maxResults=page_size, fields=fields
        )
        if not batch:
            break
        for it in batch:
            if require_components:
                comps = getattr(it.fields, "components", None) or []
                if not comps:
                    continue
            results.append(it)
        start_at += len(batch)
        if len(batch) < page_size:
            break
    return results


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_issue_keys_csv(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # CSV of keys, allow newlines/spaces
    parts = [p.strip() for p in content.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]

def main() -> None:
    jira = connect_jira()
    issue_keys = load_issue_keys_csv("v_data/representatives.txt")
    if not issue_keys:
        print("No issue keys found in v_data/representatives.txt")
        return

    all_issues: List[JiraIssue] = []
    # JQL IN list practical limits vary; keep batches small (e.g., 200)
    for key_batch in chunk_list(issue_keys, 200):
        keys_csv = ", ".join(key_batch)
        jql = f"key in ({keys_csv})"
        batch_issues = fetch_done_issues(
            jira,
            jql,
            fields=DEFAULT_FIELDS,
            require_components=True,
        )
        all_issues.extend(batch_issues)

    done_rows = [issue_to_row(i) for i in all_issues]
    save_jsonl("v_data/done_representatives.jsonl", done_rows)
    print(f"Saved {len(done_rows)} issues to done_representatives.jsonl from {len(issue_keys)} requested keys")
    if done_rows[:1]:
        print("Sample:", json.dumps(done_rows[0], ensure_ascii=False)[:300])

if __name__ == "__main__":
    main()



