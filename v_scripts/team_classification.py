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
# CONFIG (edit as needed)
# =========================
# GitLab private instance (you already used it)
GITLAB_URL   = os.environ["GITLAB_URL"]
GITLAB_TOKEN = os.environ["GITLAB_TOKEN"]
PROJECT_PATH = "idf/idf-components-mapping"
FILE_PATH    = "components.yaml"
REF          = "main"
REQUEST_TIMEOUT = 30

# Data
TEST_PATH        = "v_data/test_done_issues.jsonl"  # issues with components field (true components)
HINTS_PATH       = "v_data/team_hints.json"         # keyword -> team mapping              
OUT_PREDS_PATH   = "predictions_teams.json"

# Model
MODEL            = os.getenv("OPENAI_MODEL", "gpt-5")
TEST_LIMIT       = 200
SHOW_DIAGNOSTICS = True

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
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
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
   - **COMPONENT-FIRST RULE**: Always prioritize the component field over error descriptions
   
   **Component-to-Team Mapping**:
   - BLE/nimble/bluedroid components → BLE team (even if mentions "BT" or "Bluetooth")
   - Bluetooth Classic/Coexistence components → Classic Bluetooth team
   - Build System component (internal) → IDF Core (FreeRTOS, system core)
   - Build System tools (cmake, idf.py) → IDF Tools (developer tooling)
   - toolchain component → Toolchains & Debuggers (GCC/LLVM)
   - debugging and tracing component → Toolchains & Debuggers
   - soc/hal/driver_* components → Chip Support (peripheral drivers)
   - ULP/cxx/freertos/heap/log components → IDF Core (OS & system core)
   - modbus/LWIP/esp_netif/mdns components → Networking and Protocols
   - nvs_flash/fatfs/spiffs/vfs components → Storage
   - mbedtls/esp_tls/provisioning components → Application Utilities
   - Wi-Fi/PHY/wpa_supplicant components → Wi-Fi team
   - usb_device/usb_host components → USB team
   - tools/idf_monitor components → IDF Tools
   
   **Context Rules**:
   - NimBLE stack → BLE (not Classic Bluetooth)
   - Component resolution errors → focus on component domain, not build error
   - Driver issues → Chip Support (peripheral drivers & HAL)
   - Network protocols → Networking and Protocols
   - Security/crypto → Application Utilities or Security

Only choose from the provided team names (case-sensitive).
Provide clear technical reasoning for your choice."""

def team_schema(team_names: List[str]) -> Dict[str, Any]:
    """Schema for /v1/responses endpoint"""
    return {
        "type": "json_schema",
        "name": "assign_team",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "minItems": 1, "maxItems": 1,
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

def make_prompt_for_issue(
    team_keywords: Dict[str, List[str]],
    team_to_components: Dict[str, List[str]],
    issue: IssueRow
) -> str:
    """
    Enhanced prompt with detailed team descriptions and classification examples.
    """
    lines = []
    lines.append("Analyze this ESP-IDF issue and assign the responsible team using the technical analysis framework.\n")
    
    lines.append("AVAILABLE TEAMS:\n")
    
    # Enhanced team descriptions with clear domains
    team_descriptions = {
        "Chip Support": "Peripheral drivers & HAL (ADC/DAC/GPIO/I2C/SPI/UART/Timers), chip bring-up, hardware support (CACHE/MMU/PSRAM/FLASH), camera/LCD/sensors. Components: driver_*, soc, hal, esp_lcd, spi_flash",
        "Wi-Fi": "WiFi connectivity, esp_wifi_* APIs, PHY layer, WPA supplicant. Components: Wi-Fi, PHY, wpa_supplicant",
        "BLE": "Bluetooth Low Energy stack, GATT, GAP, NimBLE, Bluedroid BLE parts. Components: BLE, BLE_Mesh, nimble, bluedroid - ANY BLE component",
        "Classic Bluetooth": "Classic Bluetooth protocols, SPP, A2DP, HID, L2CAP, coexistence. Components: Bluetooth Classic, Coexistence - ONLY when NOT BLE",
        "Networking and Protocols": "lwIP TCP/IP stack, ethernet, protocol libraries (MQTT, mDNS, WebSocket), Modbus. Components: LWIP, esp_netif, ethernet, MQTT, mdns, modbus",
        "Application Utilities": "HTTP server/client, TLS/SSL, OTA updates, provisioning, JSON, crypto. Components: esp_http_*, esp_tls, mbedtls, app_update, provisioning",
        "Storage": "NVS flash, filesystems (FAT/SPIFFS), wear levelling, partitions, SD cards. Components: nvs_flash, fatfs, spiffs, vfs, sdmmc, storage",
        "IDF Core": "OS & system core: FreeRTOS, heap, logging, bootloader, ULP, C++ support, system features. Components: freertos, heap, log, bootloader, ULP, cxx, Build System (internal)",
        "IDF Tools": "Developer tooling: idf.py, installers, monitor, build tools, size analysis. Components: tools, idf_monitor, Build System (cmake tools)",
        "Toolchains & Debuggers": "GCC/LLVM toolchains, OpenOCD, debugging, coredump, tracing. Components: toolchain, debugging and tracing, app_trace, gdbstub, coredump",
        "USB": "USB Host/Device stacks, TinyUSB, USB OTG drivers. Components: usb_device, usb_host, usb",
        "802.15.4": "IEEE 802.15.4, Thread, Zigbee, Matter protocols. Components: 802.15.4, Thread, Zigbee, Matter",
        "Sleep and Power Management": "Deep sleep, light sleep, power management, wake sources. Components: sleep and power management",
        "Security": "Secure boot, flash encryption, cryptographic functions. Components: security, libsodium, esp_tee",
        "IDE": "IDE plugins for VS Code, Eclipse, development environment. Components: IDE",
        "Other": "Testing, CI, documentation, hardware issues, 3rd party libraries. Components: test-environments, unit_test, CI, Documentation"
    }
    
    for team, comps in team_to_components.items():
        description = team_descriptions.get(team, "")
        kws = ", ".join(team_keywords.get(team, [])[:15])  # Show more keywords
        sample_comps = ", ".join(comps[:8])  # Show more components
        
        lines.append(f"• {team}:")
        lines.append(f"  Domain: {description}")
        if kws:
            lines.append(f"  Keywords: {kws}")
        if sample_comps:
            lines.append(f"  Components: {sample_comps}")
        lines.append("")
    
    lines.append("CLASSIFICATION EXAMPLES:")
    lines.append("• 'gpio_set_level not working' → Chip Support (GPIO driver)")
    lines.append("• 'esp_wifi_connect returns ESP_ERR_WIFI_NOT_INIT' → Wi-Fi (WiFi API)")
    lines.append("• 'BLE_HS_ENOTCONN after pairing' → BLE (BLE error code)")
    lines.append("• 'ESP BLE GATT Server' + BLE component → BLE (component field is decisive)")
    lines.append("• 'BT GATT indicate' + BLE component → BLE (BLE component overrides BT mention)")
    lines.append("• 'NimBLE SPP server example' → BLE (NimBLE is BLE stack, not Classic Bluetooth)")
    lines.append("• 'Bluedroid SPP connection' → Classic Bluetooth (Bluedroid handles Classic BT)")
    lines.append("• 'nvs_flash_init fails' → Storage (NVS functionality)")
    lines.append("• 'HTTP client timeout' → Networking and Protocols (HTTP protocol)")
    lines.append("• 'Failed to resolve component mdns' → Networking and Protocols (focus on mdns component, not build error)")
    lines.append("• 'Build System component' → IDF Core (Build System is core functionality)")
    lines.append("• 'toolchain component' → Toolchains & Debuggers (not IDF Tools)")
    lines.append("• 'ULP component' → IDF Core (Ultra Low Power is core feature)")
    lines.append("• 'soc component' → Chip Support (System on Chip support)")
    lines.append("• 'idf.py build fails on Windows' → IDF Tools (build system)")
    lines.append("• 'malloc returns NULL' → IDF Core (heap management)")
    lines.append("• 'deep sleep current consumption' → Sleep and Power Management (power)")
    lines.append("")
    
    lines.append("ISSUE TO CLASSIFY:")
    issue_obj = {
        "issue_key": issue.issue_key,
        "summary": issue.summary,
        "description": issue.description
    }
    lines.append(json.dumps(issue_obj, ensure_ascii=False, indent=2))
    
    lines.append("\nAnalyze the technical signals and assign the most appropriate team.")
    return "\n".join(lines)

# =========================
# Evaluation
# =========================
def evaluate_accuracy(gold_by_key: Dict[str, str], pred_by_key: Dict[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    total = correct = 0
    mismatches = []
    
    for k, gold in gold_by_key.items():
        pred = pred_by_key.get(k)
        if not pred:
            continue
        total += 1
        if pred == gold:
            correct += 1
        else:
            mismatches.append({
                "issue_key": k,
                "predicted": pred,
                "expected": gold
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
    for it in issues:
        team = gold_team_for_issue(it.components, comp_to_team)
        if team:
            gold_by_key[it.issue_key] = team
    log.info("Gold teams available for %d issues (others skipped in eval)", len(gold_by_key))

    # 5) Predict per issue
    preds: List[Dict[str, str]] = []
    for i, it in enumerate(issues, 1):
        prompt = make_prompt_for_issue(team_keywords, team_to_components, it)
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": team_schema(team_names)
            },
        )
        # Parse response from /v1/responses format
        obj = json.loads(resp.output_text)
        pred = obj["predictions"][0]
        preds.append({"issue_key": pred["issue_key"], "team": pred["team"]})
        log.info("Pred %d/%d: %s -> %s", i, len(issues), it.issue_key, pred["team"])

    # 6) Save & evaluate
    with open(OUT_PREDS_PATH, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    log.info("Saved predictions → %s (%d)", OUT_PREDS_PATH, len(preds))

    pred_by_key = {p["issue_key"]: p["team"] for p in preds}
    metrics, mismatches = evaluate_accuracy(gold_by_key, pred_by_key)
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
