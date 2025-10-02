#!/usr/bin/env python3
"""
Build team-specific keyword hints for team assignment using:
1) Statistical mining from training data (components -> teams)
2) LLM expansion for team-specific APIs, errors, and data structures
3) Highly specific technical signals per team
"""

import os
import re
import json
import logging
import urllib.parse
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from dotenv import load_dotenv
from openai import OpenAI
import requests
import yaml

load_dotenv()

# =========================
# CONFIG
# =========================
TRAIN_PATH       = "v_data/train_done_issues.jsonl"
OUT_HINTS_PATH   = "v_data/team_hints.json"

# GitLab YAML source
GITLAB_URL   = os.environ["GITLAB_URL"]
GITLAB_TOKEN = os.environ["GITLAB_TOKEN"]
PROJECT_PATH = "idf/idf-components-mapping"
FILE_PATH    = "components.yaml"
REF          = "main"
REQUEST_TIMEOUT = 30

MODEL            = "gpt-5"
BATCH_SIZE       = 3    # teams per LLM batch
MAX_KEYWORDS     = 40   # max keywords per team from LLM

MIN_TEAM_SAMPLES = 5    # team needs >=5 training issues

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build-team-hints")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# =========================
# GitLab YAML → team mapping
# =========================
def gitlab_file_raw_url(base: str, project: str, file_path: str, ref: str) -> str:
    proj_enc = urllib.parse.quote(project, safe="")
    file_enc = urllib.parse.quote(file_path, safe="")
    return f"{base.rstrip('/')}/api/v4/projects/{proj_enc}/repository/files/{file_enc}/raw?ref={urllib.parse.quote(ref, safe='')}"

def fetch_components_yaml() -> Dict[str, Any]:
    url = gitlab_file_raw_url(GITLAB_URL, PROJECT_PATH, FILE_PATH, REF)
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    log.info("GET %s", url)
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return yaml.safe_load(r.text) or {}

def build_team_maps(doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      team_names: list[str]
      comp_to_team: {component_name -> team_name}
    """
    teams = doc.get("teams") or []
    team_names: List[str] = []
    comp_to_team: Dict[str, str] = {}
    
    for t in teams:
        team = str(t.get("name") or "").strip()
        if not team: 
            continue
        team_names.append(team)
        
        for c in (t.get("components") or []):
            name = c.get("name") if isinstance(c, dict) else c
            if not name: continue
            name = str(name).strip()
            
            # if component appears in multiple teams - keep first occurrence, log others
            if name not in comp_to_team:
                comp_to_team[name] = team
            else:
                if comp_to_team[name] != team:
                    log.warning("Component %r appears under multiple teams: %r and %r (keeping first)",
                                name, comp_to_team[name], team)
    
    return team_names, comp_to_team

# =========================
# Text cleaning (minimal - LLM handles the analysis)
# =========================
def clean_text(text: str) -> str:
    """Basic text cleaning for LLM input"""
    if not text:
        return ""
    
    # Remove URLs, code blocks, HTML - but keep technical content
    text = re.sub(r'https?://\S+', ' [URL] ', text)
    text = re.sub(r'\{code[^}]*\}(.*?)\{code\}', r' [CODE: \1] ', text, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[[^\]]*\|[^\]]*\]', ' ', text)
    
    # Clean excessive whitespace but preserve structure
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# Data loading
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

# =========================
# Token mining for teams
# =========================
def prepare_team_issues(train_data: List[Dict], comp_to_team: Dict[str, str]) -> Dict[str, List[str]]:
    """Prepare actual issue texts for each team for LLM analysis"""
    
    # Group training data by team (via components)
    team_issues = defaultdict(list)
    for item in train_data:
        summary = item.get('summary', '')
        description = item.get('description', '')
        full_text = clean_text(f"{summary}\n{description}").strip()
        
        if not full_text:
            continue
            
        item_teams = set()
        
        # Map components to teams
        for comp in item.get('components', []):
            if comp in comp_to_team:
                item_teams.add(comp_to_team[comp])
        
        # Add full issue text to all teams (issue can belong to multiple teams)
        for team in item_teams:
            team_issues[team].append(full_text)
    
    # Filter teams with enough samples and limit to reasonable number for LLM
    filtered_team_issues = {}
    for team, issues in team_issues.items():
        if len(issues) >= MIN_TEAM_SAMPLES:
            # Sort by length (longer issues often have more technical details)
            issues_sorted = sorted(issues, key=len, reverse=True)
            # Take up to 20 most detailed issues for analysis
            filtered_team_issues[team] = issues_sorted[:20]
    
    log.info(f"Prepared issues for {len(filtered_team_issues)} teams")
    for team, issues in filtered_team_issues.items():
        log.info(f"  {team}: {len(issues)} issues")
    
    return filtered_team_issues

# =========================
# LLM pattern extraction for teams
# =========================
def extract_team_patterns_with_llm(client: OpenAI, team_issues: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Let LLM analyze actual issues and extract team-specific technical patterns"""
    
    system_prompt = """You are an ESP-IDF expert analyzing bug reports to identify team-specific technical patterns.

For each team, I'll provide actual issue texts from their domain. Analyze these issues and extract:

1. **Specific API function names** mentioned in issues (esp_wifi_connect, gpio_set_level, nvs_get_blob, etc.)
2. **Specific error codes and messages** (ESP_ERR_WIFI_NOT_INIT, BLE_HS_ENOTCONN, "connection timeout", etc.)
3. **Data structures and types** (wifi_config_t, i2c_config_t, ble_gattc_char_t, etc.)
4. **Hardware/protocol-specific terms** (GPIO21, SDA/SCL, GATT, L2CAP, TCP socket, etc.)
5. **File paths and component references** (components/esp_wifi, driver/gpio, etc.)
6. **Technical concepts unique to this team** (provisioning, pairing, wear levelling, etc.)

Extract 25-40 highly specific technical keywords that are ACTUALLY mentioned in the provided issues.
Do NOT invent keywords - only extract what you see in the issue texts.
Return lowercase keywords, focus on technical specificity."""

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "extract_team_patterns",
            "schema": {
                "type": "object",
                "properties": {
                    "teams": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "team": {"type": "string"},
                                "patterns": {
                                    "type": "object",
                                    "properties": {
                                        "api_functions": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Specific API function names found in issues"
                                        },
                                        "error_codes": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Specific error codes and error messages"
                                        },
                                        "data_structures": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Data types and structures"
                                        },
                                        "technical_terms": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Hardware, protocol, and domain-specific terms"
                                        }
                                    },
                                    "required": ["api_functions", "error_codes", "data_structures", "technical_terms"]
                                }
                            },
                            "required": ["team", "patterns"]
                        }
                    }
                },
                "required": ["teams"]
            }
        }
    }
    
    extracted_patterns = {}
    teams = list(team_issues.keys())
    
    # Process in batches
    for i in range(0, len(teams), BATCH_SIZE):
        batch_teams = teams[i:i+BATCH_SIZE]
        
        # Build prompt with actual issue texts
        prompt_parts = ["Analyze these ESP-IDF team issues and extract technical patterns:\n"]
        for team in batch_teams:
            issues = team_issues[team][:10]  # Use up to 10 representative issues
            prompt_parts.append(f"\n=== {team} Team Issues ===")
            for j, issue_text in enumerate(issues, 1):
                # Truncate very long issues
                truncated = issue_text[:800] + "..." if len(issue_text) > 800 else issue_text
                prompt_parts.append(f"Issue {j}: {truncated}")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=schema,
                temperature=0
            )
            
            data = json.loads(response.choices[0].message.content)
            for item in data.get("teams", []):
                team = item["team"]
                patterns = item.get("patterns", {})
                
                # Flatten all patterns into a single keyword list
                all_keywords = []
                for category in ["api_functions", "error_codes", "data_structures", "technical_terms"]:
                    keywords = [kw.strip().lower() for kw in patterns.get(category, []) if kw.strip()]
                    all_keywords.extend(keywords)
                
                extracted_patterns[team] = all_keywords
                log.info(f"Extracted {len(all_keywords)} patterns for {team}")
                
            log.info(f"Processed batch {i//BATCH_SIZE + 1}/{(len(teams)-1)//BATCH_SIZE + 1}")
            
        except Exception as e:
            log.warning(f"Pattern extraction failed for batch {i//BATCH_SIZE + 1}: {e}")
    
    return extracted_patterns

# =========================
# Build final team hints
# =========================
def build_final_team_hints(team_names: List[str], 
                          extracted_patterns: Dict[str, List[str]]) -> Dict[str, str]:
    """Build final keyword -> team mapping from extracted patterns"""
    
    keyword_to_team = {}
    
    # Add extracted patterns
    for team, patterns in extracted_patterns.items():
        for pattern in patterns:
            if not pattern or len(pattern) < 2:
                continue
                
            keyword_to_team[pattern] = team
            
            # Add variants for technical terms
            if '_' in pattern:
                keyword_to_team[pattern.replace('_', '')] = team
                keyword_to_team[pattern.replace('_', ' ')] = team
            
            # Handle common suffixes
            if pattern.endswith('_t'):
                keyword_to_team[pattern[:-2]] = team
            elif pattern.endswith('_config'):
                keyword_to_team[pattern[:-7]] = team
    
    return keyword_to_team

# =========================
# Main
# =========================
def main():
    """Main function"""
    log.info("Building team-specific keyword hints...")
    
    # Load team mapping from GitLab YAML
    doc = fetch_components_yaml()
    team_names, comp_to_team = build_team_maps(doc)
    log.info(f"Loaded: {len(team_names)} teams, {len(comp_to_team)} component mappings")
    
    # Load training data
    train_data = load_jsonl(TRAIN_PATH)
    log.info(f"Loaded: {len(train_data)} training items")
    
    # Prepare actual issue texts for LLM analysis
    team_issues = prepare_team_issues(train_data, comp_to_team)
    log.info(f"Prepared issues for {len(team_issues)} teams")
    
    # Extract patterns with LLM
    client = OpenAI(api_key=OPENAI_API_KEY)
    extracted_patterns = extract_team_patterns_with_llm(client, team_issues)
    log.info(f"LLM extracted patterns for {len(extracted_patterns)} teams")
    
    # Build final hints
    hints = build_final_team_hints(team_names, extracted_patterns)

    # Save
    os.makedirs(os.path.dirname(OUT_HINTS_PATH), exist_ok=True)
    with open(OUT_HINTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(hints, f, ensure_ascii=False, indent=2)
    
    log.info(f"Saved {len(hints)} team hints to {OUT_HINTS_PATH}")
    
    # Show summary
    team_counts = Counter(hints.values())
    print(f"\nGenerated {len(hints)} keyword hints for {len(team_counts)} teams:")
    for team, count in team_counts.most_common():
        print(f"  {team}: {count} keywords")

if __name__ == "__main__":
    main()
