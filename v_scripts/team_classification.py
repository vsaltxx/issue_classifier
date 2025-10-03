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
TEST_LIMIT       = 500
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


def normalize_team_name(team_name: str) -> str:
    """Normalize team name for matching"""
    # Remove "team" suffix and normalize
    normalized = team_name.lower().strip()
    if normalized.endswith(" team"):
        normalized = normalized[:-5].strip()
    return normalized

def build_team_keywords(
    team_to_components: Dict[str, List[str]],
    hints_kw2team: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Convert keyword->team mapping to team->keywords mapping.
    Uses team_hints.json data to build keyword lists for each team.
    """
    per_team: Dict[str, List[str]] = defaultdict(list)
    
    # Create normalized team name mapping
    gitlab_teams_normalized = {}
    for team in team_to_components.keys():
        normalized = normalize_team_name(team)
        gitlab_teams_normalized[normalized] = team
    
    log.info("GitLab teams normalized: %s", list(gitlab_teams_normalized.keys()))
    
    # Convert keyword->team to team->keywords
    matched_teams = set()
    unmatched_teams = set()
    
    for keyword, team_name in hints_kw2team.items():
        # Normalize team name from hints
        team_name_normalized = normalize_team_name(team_name)
        
        # Find matching team
        matching_team = gitlab_teams_normalized.get(team_name_normalized)
        
        if matching_team:
            per_team[matching_team].append(keyword.strip())
            matched_teams.add(team_name)
        else:
            unmatched_teams.add(team_name)
    
    log.info("Matched %d hint teams, unmatched: %d", len(matched_teams), len(unmatched_teams))
    if unmatched_teams:
        log.info("Unmatched hint teams: %s", sorted(list(unmatched_teams))[:5])
    
    # Clean and deduplicate keywords per team
    for team in per_team:
        keywords = per_team[team]
        # Remove duplicates while preserving order
        seen = set()
        filtered = []
        for kw in keywords:
            kw_clean = str(kw).strip()
            if not kw_clean or len(kw_clean) < 2: 
                continue
            if kw_clean.lower() in seen:
                continue
            seen.add(kw_clean.lower())
            filtered.append(kw_clean)
        per_team[team] = filtered
    
    # Ensure all teams have at least an empty list
    for team in team_to_components:
        if team not in per_team:
            per_team[team] = []
    
    teams_with_keywords = len([t for t in per_team if per_team[t]])
    total_keywords = sum(len(kws) for kws in per_team.values())
    log.info("Built keywords for %d/%d teams, total keywords: %d", 
             teams_with_keywords, len(team_to_components), total_keywords)
    
    
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
SYSTEM_MSG = """You are an ESP-IDF issue classifier. Assign exactly ONE team (from CandidateTeams) based on technical signals.

HARD ANTI-"Other" GUARDRAIL:
NEVER choose "Other" if ANY of these ESP-IDF tokens appear (case-insensitive, word-boundary):
esp_*, ble_*, nimble, gatt, lwip, mdns, mqtt, esp_netif, http_client, http_server, ota, twai, spi, i2c, i2s, uart, rmt, usb, sdmmc, spiffs, fatfs, nvs, esp_pm, esp_sleep, bootloader, partition_table

EXPLICIT OVERRIDES (apply before general rules):

A) Application Utilities vs Networking & Protocols:
→ Application Utilities: esp_http_client, esp_https_ota, esp_tls (with OTA), esp_http_server, httpd_ws, provisioning, protocomm, wifi_prov_mgr
→ Networking & Protocols: esp_mqtt, esp_websocket_client, esp_netif, lwip, PPP, mdns, esp_modem

B) IDF Tools vs IDE:
→ IDE only if: "Build, Flash and Monitor button", "ESP-IDF: Doctor", extension settings, .vscode/launch.json, IntelliSense/indexer
→ IDF Tools: default even if VS Code mentioned

C) IDF Core vs IDF Tools:
→ IDF Core: idf_component_register, CMakeLists.txt, linker script, Guru Meditation, panic, esp_system, bootloader, partition_table, settimeofday, newlib, heap_caps, freertos
→ IDF Tools: install.sh/.ps1, offline installer, idf_tools.py, OpenOCD/libusb, Python venv, idf.py UX

D) Sleep & Power vs Chip Support:
→ Sleep & Power: esp_sleep_*, esp_pm_*, deep sleep, light sleep, DFS, wakeup, ULP, RTC GPIO, APB/CPU freq
→ Chip Support: peripheral drivers (SPI/I2C/UART/I2S/RMT/ADC/LCD/SD) even with watchdog

E) USB cases:
→ USB: usb_device/usb_host/TinyUSB
→ IDF Core: usb_serial_jtag console, CDC console, console reboot

F) BLE vs Classic:
→ BLE: GATT, NimBLE, BLE_MESH, esp_blufi_*, protocomm_nimble
→ Classic: SPP, A2DP, AVRCP, HFP, HID/HIDH, RFCOMM, esp_bt_gap_*

G) Wi-Fi vs Networking:
→ Wi-Fi: esp_wifi_* APIs, driver issues
→ Networking: esp_netif, lwip, mdns, sockets, PPP, protocol clients

H) Storage/VFS:
→ Chip Support: VFS for UART (esp_vfs_dev, line endings)
→ IDF Core: partition table, boot image
→ IDF Tools: filesystem tools (generate/preprogram images)

CLASSIFICATION PROCESS:
1. Apply EXPLICIT OVERRIDES first (A-H above)
2. Check API prefixes: esp_ble_*/ble_*/nimble → BLE; esp_wifi_* → Wi-Fi; gpio_*/i2c_*/spi_*/uart_* → Chip Support; etc.
3. Look for component paths and build tokens
4. Use domain context as fallback

SPECIAL CASES:
• ESP32-C2/C3/C6/S3/H2 + Bluetooth → BLE (no Classic support)
• "NimBLE SPP" → BLE
• Component manager dependency issues → IDF Tools; runtime protocol behavior → Networking

OUTPUT (JSON only):
{"predictions":[{"issue_key":"<key>","team":"<exact team name>","reasoning":"Brief rationale with key signals"}]}
"""



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
    """Prompt with component mappings, keywords, and contrastive examples."""
    
    # Build team-component mapping
    team_components_text = ""
    for team, components in team_to_components.items():
        components_list = ", ".join(components)
        team_components_text += f"• {team}: {components_list}\n"
    
    # Build team keywords mapping (condensed)
    team_keywords_text = ""
    for team, keywords in team_keywords.items():
        if keywords:  # Only show teams that have keywords
            # Limit to first 10 keywords to keep prompt manageable
            keywords_display = keywords[:10]
            if len(keywords) > 10:
                keywords_display.append(f"... (+{len(keywords)-10} more)")
            keywords_list = ", ".join(keywords_display)
            team_keywords_text += f"• {team}: {keywords_list}\n"
    
    # Contrastive few-shot examples
    examples_text = """CONTRASTIVE EXAMPLES:

**IDF Tools vs IDE:**
• "idf.py build fails with CMake error" + Terminal usage → IDF Tools
• "ESP-IDF VS Code extension setup failing" → IDE (extension issue)
• "Build, Flash and Monitor button fails" + VS Code extension → IDE

**IDF Core vs IDF Tools:**
• "CMakeLists.txt: idf_component_register() missing argument" → IDF Core  
• "install.sh fails to download toolchain" → IDF Tools

**BLE vs Classic Bluetooth:**
• "GATT server MTU negotiation fails" + esp_ble_gatts_* → BLE
• "Missing esp_bt_defs.h in BT component" → BLE (shared BT headers)
• "A2DP audio streaming drops connection" + esp_a2d_* → Classic Bluetooth
• "NimBLE SPP server example doesn't work" → BLE (SPP over BLE)

**Sleep and Power Management vs Chip Support:**
• "esp_deep_sleep_start() doesn't wake on GPIO" → Sleep and Power Management
• "gpio_wakeup_enable() not working" → Chip Support (GPIO driver API)
• "DFS affecting LEDC frequency" → Chip Support (peripheral driver issue)
• "Power consumption high in light sleep" → Sleep and Power Management

**Networking and Protocols vs IDF Tools:**
• "MQTT client disconnects randomly" + esp_mqtt_client_* → Networking and Protocols
• "Component manager can't resolve 'mdns' dependency" → Networking and Protocols
• "idf.py monitor crashes when device sends data" → IDF Tools

**Toolchains & Debuggers vs IDF Tools:**
• "Request M1-native toolchain support" → Toolchains & Debuggers
• "Question about C/C++ language standards used by IDF" → Toolchains & Debuggers
• "Integrate clangd v19 into tools.json" → Toolchains & Debuggers
• "idf.py build fails with CMake error" → IDF Tools

**IDF Core vs IDF Tools (Build System):**
• "Reproducible builds affected by esptool.py" → IDF Core (build system internals)
• "Builds failing offline due to dependency fetching" → IDF Core (build system)
• "Undocumented change causing build failures" → IDF Core (API/build changes)
• "idf.py command not found" → IDF Tools (tool installation)

**IDF Core vs Other (Documentation):**
• "Typo in documentation fixture docstring" → Other
• "Misleading function description in docs" → Other  
• "Request to document known broken items" → Other

**Other vs Component Teams:**
• "Question about using external library (iconv) with ESP-IDF" → IDF Core (integration)
• "Android ANCS notifications guidance" → Other (pure guidance)
• "Misleading API description for specific component" → Chip Support (component docs)
• "HTTP reboot feature request for ESP-IDF" → Application Utilities (HTTP features)
• "Component path moved in minor release" → Networking and Protocols (component affected)
• "TWAI driver TX/RX issues" → Chip Support (driver functionality)

**USB vs Toolchains & Debuggers:**
• "Wrong JTAG USB PID under USB bridge" → USB (USB device/descriptor)
• "GDB debugging over JTAG" → Toolchains & Debuggers

**Wi-Fi vs Networking:**
• "esp_wifi_connect() returns ESP_FAIL" → Wi-Fi
• "HTTP client SSL handshake fails" + esp_http_client_* → Networking and Protocols"""
    
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
    
    prompt_parts = [
        issue_header,
        "",
        "TEAM COMPONENT MAPPINGS:",
        team_components_text,
    ]
    
    # Add team keywords section if we have any
    if team_keywords_text.strip():
        prompt_parts.extend([
            "TEAM TECHNICAL KEYWORDS:",
            team_keywords_text,
        ])
    
    prompt_parts.extend([
        examples_text,
        "",
        "ISSUES:",
        issues_text
    ])
    
    return "\n".join(prompt_parts)


# =========================
# Evaluation
# =========================
def evaluate_accuracy(gold_by_key: Dict[str, str], pred_by_key: Dict[str, str], gold_teams_by_key: Dict[str, List[str]] = None, pred_reasoning_by_key: Dict[str, str] = None) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
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
            
            mismatch_data = {
                "issue_key": k,
                "predicted": pred,
                "expected": expected_display
            }
            
            # Add reasoning if available
            if pred_reasoning_by_key and k in pred_reasoning_by_key:
                mismatch_data["reasoning"] = pred_reasoning_by_key[k]
            
            mismatches.append(mismatch_data)
    
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
            issue_key = pred["issue_key"]
            predicted_team = pred["team"]
            reasoning = pred.get("reasoning", "No reasoning provided")
            
            # Store prediction with reasoning
            preds.append({
                "issue_key": issue_key, 
                "team": predicted_team,
                "reasoning": reasoning
            })
            
            # Simple logging for all predictions
            log.info("Pred: %s -> %s", issue_key, predicted_team)

    # 6) Save & evaluate
    with open(OUT_PREDS_PATH, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    log.info("Saved predictions → %s (%d)", OUT_PREDS_PATH, len(preds))

    pred_by_key = {p["issue_key"]: p["team"] for p in preds}
    pred_reasoning_by_key = {p["issue_key"]: p.get("reasoning", "No reasoning provided") for p in preds}
    metrics, mismatches = evaluate_accuracy(gold_by_key, pred_by_key, gold_teams_by_key, pred_reasoning_by_key)
    log.info("Evaluation: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    
    # Show detailed mismatches
    if mismatches:
        print("\n" + "="*60)
        print(f"MISMATCHED CLASSIFICATIONS ({len(mismatches)} issues)")
        print("="*60)
        
        for i, mismatch in enumerate(mismatches, 1):
            key = mismatch["issue_key"]
            
            # Enhanced logging for mismatched cases only
            log.info("MISMATCH - Key: %s | Expected: %s | Predicted: %s | Reasoning: %s", 
                    key, mismatch['expected'], mismatch['predicted'], 
                    mismatch.get('reasoning', 'No reasoning provided'))
        
        print("\n" + "="*60)
    else:
        print("\nNo mismatches - perfect accuracy!")

if __name__ == "__main__":
    main()
