#!/usr/bin/env python3
import os
import json
import logging
import urllib.parse
from typing import Any, Dict, List

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()
# ========= CONFIG =========
GITLAB_URL      = os.environ["GITLAB_URL"]
GITLAB_TOKEN    = os.environ["GITLAB_TOKEN"]      
PROJECT_PATH    = "idf/idf-components-mapping"    # <group>/<project>
FILE_PATH       = "components.yaml"
REF             = "main"
OUTPUT_PATH     = "components.json"               # list[str] of component names
REQUEST_TIMEOUT = 30


# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("components-loader")


def gitlab_file_raw_url(base: str, project: str, file_path: str, ref: str) -> str:
    """Build the GitLab API v4 raw file URL (works for private projects with token)."""
    proj_enc = urllib.parse.quote(project, safe="")
    file_enc = urllib.parse.quote(file_path, safe="")
    return f"{base.rstrip('/')}/api/v4/projects/{proj_enc}/repository/files/{file_enc}/raw?ref={urllib.parse.quote(ref, safe='')}"


def fetch_yaml_text() -> str:
    if not GITLAB_TOKEN:
        raise ValueError("GITLAB_TOKEN environment variable is required")
    url = gitlab_file_raw_url(GITLAB_URL, PROJECT_PATH, FILE_PATH, REF)
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    log.info("GET %s", url)
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def extract_components(doc: Dict[str, Any]) -> List[str]:
    """
    Extract component names from teams structure:
    { "teams": [ {"name": "...", "components": [{"name": "CompA"}, ...]}, ... ] }
    Returns a deduplicated, sorted list of component names.
    """
    names: List[str] = []

    # Process teams structure
    if isinstance(doc.get("teams"), list):
        for team in doc["teams"]:
            components = team.get("components") or []
            for component in components:
                if isinstance(component, dict):
                    name = component.get("name")
                    if name:
                        names.append(str(name).strip())

    # De-duplicate preserving order
    seen = set()
    deduped = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)

    # Sort for stability (optional). Comment out if you want original order.
    deduped.sort(key=str.lower)
    return deduped


def main() -> None:
    try:
        yaml_text = fetch_yaml_text()
        doc = yaml.safe_load(yaml_text) or {}
        components = extract_components(doc)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(components, f, ensure_ascii=False, indent=2)
        log.info("Saved %d components → %s", len(components), OUTPUT_PATH)
        log.info("Sample: %s", components[:5])
    except requests.HTTPError as e:
        log.error("HTTP error: %s\nResponse: %s", e, getattr(e.response, "text", "")[:500])
        raise


if __name__ == "__main__":
    main()
