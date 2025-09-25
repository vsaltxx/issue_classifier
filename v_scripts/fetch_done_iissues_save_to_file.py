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

def main() -> None:
    jira = connect_jira()
    issues = fetch_done_issues(jira, f"project = {PROJECT_KEY} AND status = Done AND summary ~ 'GH *'", fields=DEFAULT_FIELDS, require_components=True)
    done_rows = [issue_to_row(i) for i in issues]
    save_jsonl("done_issues.jsonl", done_rows)
    print(f"Saved {len(done_rows)} issues to done_issues.jsonl")
    if done_rows[:1]:
        print("Sample:", json.dumps(done_rows[0], ensure_ascii=False)[:300])

if __name__ == "__main__":
    main()



