#!/usr/bin/env python3
"""
Classify GitHub/Jira issues to a TEAM.
- Loads teams/components mapping from GitLab YAML
- Aggregates keywords per team
- Predicts exactly one team per issue via LLM (JSON schema enum)
- Evaluates accuracy by mapping gold components -> team
"""

import os, json, re, logging, urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import requests, yaml
from collections import defaultdict, Counter
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
# =========================
# CONFIG
# =========================
GITLAB_URL   = os.environ["GITLAB_URL"]
GITLAB_TOKEN = os.environ["GITLAB_TOKEN"]
PROJECT_PATH = "idf/idf-components-mapping"
FILE_PATH    = "components.yaml"
REF          = "main"

# Data
TEST_PATH        = "v_data/test_done_issues.jsonl"  # issues with components field (true components)
HINTS_PATH       = "v_data/team_hints.json"         # keyword -> team mapping              
OUT_PREDS_PATH   = "predictions_teams.json"

# Model
MODEL            = os.getenv("OPENAI_MODEL", "gpt-5")
TEST_LIMIT       = 100
SHOW_DIAGNOSTICS = True
BATCH_SIZE       = 20  # Process N issues at once

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("team-classifier")

# =========================
# Utils: text cleaning
# =========================
_CLEAN_LINKS = re.compile(r'https?://\S+')
_CLEAN_GH    = re.compile(r'\[GitHub Issue\|[^\]]+\]')
_CLEAN_CODE  = re.compile(r'\{code\}.*?\{code\}', flags=re.S)

def clean_text(s: str, max_len: int | None = 600) -> str:
    if not s: return ""
    s = _CLEAN_GH.sub(" ", s)
    s = _CLEAN_LINKS.sub(" ", s)
    s = _CLEAN_CODE.sub(" ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:max_len] if max_len else s

# =========================
# Data structures
# =========================
@dataclass
class IssueRow:
    issue_key: str
    summary: str
    description: str
    components: List[str]  # gold component labels (can be multiple)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def normalize_issue(raw: Dict[str, Any]) -> IssueRow:
    key = (raw.get("issue_key") or raw.get("key") or "").strip()
    summary = clean_text(raw.get("summary") or "", 200)
    desc    = clean_text(raw.get("description") or "", 800)
    comps   = [str(c).strip() for c in (raw.get("components") or []) if str(c).strip()]
    return IssueRow(issue_key=key, summary=summary, description=desc, components=comps)

# =========================
# GitLab YAML → mapping
# =========================
def gitlab_file_raw_url(base: str, project: str, file_path: str, ref: str) -> str:
    proj_enc = urllib.parse.quote(project, safe="")
    file_enc = urllib.parse.quote(file_path, safe="")
    return f"{base.rstrip('/')}/api/v4/projects/{proj_enc}/repository/files/{file_enc}/raw?ref={urllib.parse.quote(ref, safe='')}"

def fetch_components_yaml() -> Dict[str, Any]:
    url = gitlab_file_raw_url(GITLAB_URL, PROJECT_PATH, FILE_PATH, REF)
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    log.info("GET %s", url)
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return yaml.safe_load(r.text) or {}

def build_team_maps(doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
      team_names: list[str]
      comp_to_team: {component_name -> team_name}
      team_to_components: {team_name -> [component_name, ...]}
    """
    teams = doc.get("teams") or []
    team_names: List[str] = []
    comp_to_team: Dict[str, str] = {}
    team_to_components: Dict[str, List[str]] = {}
    for t in teams:
        team = str(t.get("name") or "").strip()
        if not team: 
            continue
        team_names.append(team)
        comps: List[str] = []
        for c in (t.get("components") or []):
            name = c.get("name") if isinstance(c, dict) else c
            if not name: continue
            name = str(name).strip()
            comps.append(name)
            # if component appears in multiple teams - keep first occurrence, log others
            if name not in comp_to_team:
                comp_to_team[name] = team
            else:
                # duplicate - log it (this happens: vfs, bootloader, etc.)
                if comp_to_team[name] != team:
                    log.warning("Component %r appears under multiple teams: %r and %r (keeping first)",
                                name, comp_to_team[name], team)
        team_to_components[team] = comps
    return team_names, comp_to_team, team_to_components

# =========================
# Team-level keywords
# =========================
def load_hints(path: str) -> Dict[str, str]:
    """
    team_hints.json: {"keyword": "team", ...}
    """
    if not os.path.exists(path): 
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to load hints %s: %s", path, e)
        return {}


def build_team_keywords(
    team_to_components: Dict[str, List[str]],
    hints_kw2team: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Collect keywords for each team:
    - from team_hints.json (keyword -> team) directly
    """
    per_team: Dict[str, List[str]] = defaultdict(list)
    
    # Direct mapping from keyword to team
    for kw, team in hints_kw2team.items():
        if team in team_to_components:  # Only include teams we know about
            per_team[team].append(kw)
    
    # Clean and limit keywords per team
    for team in per_team:
        keywords = per_team[team]
        # Remove duplicates while preserving order
        seen = set()
        filtered = []
        for w in keywords:
            w = str(w).lower().strip()
            if not w or len(w) < 2: 
                continue
            if w in seen:
                continue
            seen.add(w)
            filtered.append(w)
        per_team[team] = filtered
    
    # Ensure all teams have at least an empty list
    for team in team_to_components:
        if team not in per_team:
            per_team[team] = []
    
    return dict(per_team)

# =========================
# Gold: components -> team
# =========================
def gold_teams_for_issue(components: List[str], comp_to_team: Dict[str, str]) -> List[str]:
    """
    Find all possible gold teams for an issue based on its components.
    Returns list of teams that have components in this issue.
    """
    if not components:
        return []
    teams = [comp_to_team.get(c) for c in components if comp_to_team.get(c)]
    return list(set(teams))  # Remove duplicates

def gold_team_for_issue(components: List[str], comp_to_team: Dict[str, str]) -> str | None:
    """
    Map issue component list to one team:
    - if all components map to the same team → that's the gold team
    - if multiple teams → take the most frequent one (by count) or None if conflicting
    """
    if not components:
        return None
    teams = [comp_to_team.get(c) for c in components if comp_to_team.get(c)]
    if not teams:
        return None
    cnt = Counter(teams)
    top_team, top_n = cnt.most_common(1)[0]
    # If there's a clear leader - take it; in case of tie, still take the first one
    return top_team

# =========================
# LLM schema/prompt
# =========================
SYSTEM_MSG = """You are an expert ESP-IDF triage assistant with deep knowledge of team responsibilities and technical domains.

Your task is to assign exactly ONE responsible team for each issue based on technical analysis.

Analysis Framework:
1. IDENTIFY technical signals in the issue:
   - API function names (esp_wifi_*, gpio_*, nvs_*, i2c_*, etc.)
   - Error codes (ESP_ERR_*, BLE_HS_*, specific error messages)
   - Component/file paths (components/esp_wifi, driver/gpio, etc.)
   - Hardware terms (GPIO pins, I2C, SPI, UART, etc.)
   - Protocol names (HTTP, MQTT, BLE, WiFi, etc.)

2. MATCH signals to team expertise:
   - **PRIMARY**: Check component field first - it's the most reliable signal
   - Look for team-specific keywords and APIs
   - Consider the primary technical domain
   - Focus on the ROOT CAUSE, not just symptoms
   - When component field conflicts with description, trust the component

3. APPLY classification rules:
   - **COMPONENT-FIRST RULE (ENHANCED)**: ALWAYS prioritize the component field over issue description content
     * Even if issue mentions documentation, IDE setup, or power management
     * Wi-Fi component → Wi-Fi team (regardless of content type)
     * tools component → IDF Tools team (not IDE team)
     * driver component → Chip Support team (even with power-related content)
     * Only analyze description content when component field is missing or ambiguous
   
   **Component-to-Team Mapping (UPDATED)**:
   - BLE/nimble/bluedroid components → BLE team (even if mentions "BT" or "Bluetooth")
   - Bluetooth Classic/Coexistence components → Classic Bluetooth team
   - toolchain component → Toolchains & Debuggers (GCC/LLVM)
   - debugging and tracing component → Toolchains & Debuggers (not USB team)
   - soc/hal/driver_*/driver components → Chip Support (peripheral drivers)
   - usb_serial_jtag component → Chip Support (serial interface driver, not USB team)
   - ULP/cxx/freertos/heap/log components → IDF Core (OS & system core)
   - Build System/IDF_Core components → IDF Core (framework usage)
   - modbus/LWIP/esp_netif/mdns components → Networking and Protocols
   - nvs_flash/fatfs/spiffs/vfs components → Storage
   - mbedtls/esp_tls/provisioning components → Application Utilities
   - Wi-Fi/PHY/wpa_supplicant components → Wi-Fi team
   - usb_device/usb_host components → USB team (USB protocol stack)
   - tools/idf_monitor components → IDF Tools (not IDE team)
   
   **Build System Disambiguation (CRITICAL)**:
   Route to **IDF Core** when:
   - Component is "Build System", "IDF_Core", or "cxx"
   - Issue is about framework usage patterns, component integration, or language standards
   - Problems with user component structure, CMakeLists.txt usage, or Kconfig integration
   - Reproducible builds, linking issues, or build configuration problems
   - Language standard support (C++, C features)
   
   Route to **IDF Tools** when:
   - Component is "tools"
   - Error mentions tools/cmake/*.cmake files or build system internals
   - VS Code extension, idf.py, or development environment setup issues
   - CMake engine failures during configure/generate phase
   
   **Driver vs Power Management**:
   - If component is "driver" or "driver_*" → Always route to Chip Support
   - Only route to Sleep and Power Management for components: "sleep and power management", "pm", "esp_pm"
   - DFS, frequency scaling in driver context → Chip Support (driver behavior)
   - DFS, frequency scaling in power context → Sleep and Power Management
   
   **USB vs Serial Disambiguation**:
   - usb_serial_jtag component → Chip Support (serial interface driver)
   - usb_device/usb_host components → USB team (USB protocol stack)
   - debugging and tracing with USB mention → Toolchains & Debuggers (debug tools)
   
   **Context Rules**:
   - NimBLE stack → BLE (not Classic Bluetooth)
   - Component resolution errors → focus on component domain, not build error
   - Driver issues → Chip Support (peripheral drivers & HAL)
   - Network protocols → Networking and Protocols
   - Security/crypto → Application Utilities or Security

Only choose from the provided team names (case-sensitive).
Provide clear technical reasoning for your choice."""

def team_schema(team_names: List[str], batch_size: int = 1) -> Dict[str, Any]:
    """Schema for /v1/responses endpoint"""
    return {
        "type": "json_schema",
        "name": "assign_team",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "minItems": 1, "maxItems": batch_size,
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_key": {"type": "string"},
                            "team": {"type": "string", "enum": team_names},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["issue_key", "team", "reasoning"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["predictions"],
            "additionalProperties": False
        },
        "strict": True
    }


def make_prompt(
    team_keywords: Dict[str, List[str]],
    team_to_components: Dict[str, List[str]],
    issues: List[IssueRow]
) -> str:
    """Simple prompt with component mappings."""
    
    # Build team-component mapping
    team_components_text = ""
    for team, components in team_to_components.items():
        components_list = ", ".join(components)
        team_components_text += f"• {team}: {components_list}\n"
    
    # Build issues text
    if len(issues) == 1:
        issue_header = "Analyze this ESP-IDF issue and assign the responsible team."
        issues_text = json.dumps({
            "issue_key": issues[0].issue_key,
            "summary": issues[0].summary,
            "description": issues[0].description
        }, ensure_ascii=False, indent=2)
    else:
        issue_header = f"Analyze these {len(issues)} ESP-IDF issues and assign the responsible team for each."
        issues_list = []
        for i, issue in enumerate(issues, 1):
            issue_obj = {
                "issue_key": issue.issue_key,
                "summary": issue.summary,
                "description": issue.description
            }
            issues_list.append(f"Issue {i}:\n{json.dumps(issue_obj, ensure_ascii=False, indent=2)}")
        issues_text = "\n\n".join(issues_list)
    
    return f"""
{issue_header}

TEAM COMPONENT MAPPINGS:
{team_components_text}

CLASSIFICATION RULES:
• Use component field as primary signal (most reliable)
• If no component, analyze technical content for APIs, error codes, or domains
• Build system: Framework usage → IDF Core, Tool internals → IDF Tools

ISSUES:
{issues_text}
""".strip()


# =========================
# Evaluation
# =========================
def evaluate_accuracy(gold_by_key: Dict[str, str], pred_by_key: Dict[str, str], gold_teams_by_key: Dict[str, List[str]] = None) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Evaluate accuracy with multi-component support.
    If gold_teams_by_key is provided, count as correct if predicted team is in any of the gold teams.
    """
    total = correct = 0
    mismatches = []
    
    for k, gold in gold_by_key.items():
        pred = pred_by_key.get(k)
        if not pred:
            continue
        total += 1
        
        # Check if prediction is correct
        is_correct = False
        if pred == gold:
            is_correct = True
        elif gold_teams_by_key and k in gold_teams_by_key:
            # Multi-component case: check if predicted team is in any of the valid teams
            valid_teams = gold_teams_by_key[k]
            if pred in valid_teams:
                is_correct = True
        
        if is_correct:
            correct += 1
        else:
            expected_display = gold
            if gold_teams_by_key and k in gold_teams_by_key and len(gold_teams_by_key[k]) > 1:
                expected_display = f"{gold} (or {', '.join(gold_teams_by_key[k])})"
            
            mismatches.append({
                "issue_key": k,
                "predicted": pred,
                "expected": expected_display
            })
    
    acc = (correct / total) if total else 0.0
    metrics = {"evaluated": total, "correct": correct, "accuracy": acc}
    return metrics, mismatches

# =========================
# Main
# =========================
def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 1) Mapping from GitLab YAML
    doc = fetch_components_yaml()
    team_names, comp_to_team, team_to_components = build_team_maps(doc)
    log.info("Teams=%d, components mapped=%d", len(team_names), len(comp_to_team))

    # 2) Team keywords (team_hints.json)
    hints_kw2team = load_hints(HINTS_PATH)
    team_keywords = build_team_keywords(team_to_components, hints_kw2team)
    log.info("Built team keywords for %d teams", len(team_keywords))

    # 3) Load test issues and normalize
    raw_test = read_jsonl(TEST_PATH)
    issues = [normalize_issue(r) for r in raw_test[:TEST_LIMIT]]
    log.info("Loaded test issues: %d", len(issues))

    # 4) Gold teams by mapping components -> team
    gold_by_key: Dict[str, str] = {}
    gold_teams_by_key: Dict[str, List[str]] = {}
    for it in issues:
        team = gold_team_for_issue(it.components, comp_to_team)
        teams = gold_teams_for_issue(it.components, comp_to_team)
        if team:
            gold_by_key[it.issue_key] = team
            gold_teams_by_key[it.issue_key] = teams
    log.info("Gold teams available for %d issues (others skipped in eval)", len(gold_by_key))

    # 5) Predict per issue (with batching for efficiency)
    preds: List[Dict[str, str]] = []
    
    for batch_start in range(0, len(issues), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(issues))
        batch_issues = issues[batch_start:batch_end]
        
        log.info("Processing batch %d-%d/%d", batch_start + 1, batch_end, len(issues))
        
        prompt = make_prompt(team_keywords, team_to_components, batch_issues)
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": team_schema(team_names, len(batch_issues))
            },
        )
        # Parse response from /v1/responses format
        obj = json.loads(resp.output_text)
        batch_preds = obj["predictions"]
        
        for pred in batch_preds:
            preds.append({"issue_key": pred["issue_key"], "team": pred["team"]})
            log.info("Pred: %s -> %s", pred["issue_key"], pred["team"])

    # 6) Save & evaluate
    with open(OUT_PREDS_PATH, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    log.info("Saved predictions → %s (%d)", OUT_PREDS_PATH, len(preds))

    pred_by_key = {p["issue_key"]: p["team"] for p in preds}
    metrics, mismatches = evaluate_accuracy(gold_by_key, pred_by_key, gold_teams_by_key)
    log.info("Evaluation: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    
    # Show detailed mismatches
    if mismatches:
        print("\n" + "="*60)
        print(f"MISMATCHED CLASSIFICATIONS ({len(mismatches)} issues)")
        print("="*60)
        
        # Find the original issue data for context
        issue_lookup = {issue.issue_key: issue for issue in issues}
        
        for i, mismatch in enumerate(mismatches, 1):
            key = mismatch["issue_key"]
            issue = issue_lookup.get(key)
            
            print(f"\n{i}. Issue: {key}")
            print(f"   Predicted: {mismatch['predicted']}")
            print(f"   Expected:  {mismatch['expected']}")
            
            if issue:
                print(f"   Summary:   {issue.summary[:100]}{'...' if len(issue.summary) > 100 else ''}")
                print(f"   Components: {', '.join(issue.components)}")
                if issue.description:
                    desc_preview = issue.description[:150].replace('\n', ' ')
                    print(f"   Description: {desc_preview}{'...' if len(issue.description) > 150 else ''}")
        
        print("\n" + "="*60)
    else:
        print("\nNo mismatches - perfect accuracy!")

if __name__ == "__main__":
    main()
