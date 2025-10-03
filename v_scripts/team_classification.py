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
TEST_LIMIT       = 60
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
    
    # Log sample keywords for debugging
    for team, keywords in per_team.items():
        if keywords:
            log.info("Team '%s': %d keywords (sample: %s)", 
                    team, len(keywords), ", ".join(keywords[:5]))
    
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

Your task is to assign exactly ONE responsible team for each issue based on technical analysis of the issue content.

Analysis Framework:
1. IDENTIFY technical signals in the issue:
   - API function names (esp_wifi_*, gpio_*, nvs_*, i2c_*, etc.)
   - Error codes (ESP_ERR_*, BLE_HS_*, specific error messages)
   - File paths and includes (components/esp_wifi, driver/gpio, etc.)
   - Hardware terms (GPIO pins, I2C, SPI, UART, ADC, etc.)
   - Protocol names (HTTP, MQTT, BLE, WiFi, TCP, etc.)
   - Technical keywords and domain-specific terms

2. MATCH signals to team expertise:
   - Use the provided team technical keywords to identify domain matches
   - Look for specific function names, data structures, and error patterns
   - Consider the primary technical domain and root cause
   - Match issue symptoms to team specializations

3. APPLY classification rules:
   - Analyze issue summary and description for technical indicators
   - Match technical keywords, APIs, and error codes to team expertise
   - Focus on the ROOT CAUSE and primary technical domain
   - When multiple teams could apply, choose the most specific match
   
   **Domain Guidelines**:
   - WiFi APIs, connection issues → Wi-Fi team
   - Bluetooth/BLE APIs, pairing, GATT → BLE or Classic Bluetooth teams
   - GPIO, ADC, SPI, I2C drivers → Chip Support team
   - HTTP, MQTT, TCP/IP, networking → Networking and Protocols team
   - Build system, toolchain, IDE issues → IDF Tools or IDF Core teams
   - Power management, sleep modes → Sleep and Power Management team

  4. Use the following rules to distinguish IDF Core vs IDF Tools issues.

Classify as IDF Core when the issue is about code that runs on the device, ESP-IDF runtime libraries, or the ESP-IDF CMake build logic for components/projects:
- Technical indicators (APIs, symbols, headers, paths):
  - Runtime APIs: esp_event, esp_wifi, esp_log, esp_system, FreeRTOS (tasks, event groups), lwIP, newlib (settimeofday), heap/heap_caps, spi_flash, NVS, driver/*.h, esp_rom_* functions.
  - Boot and security: Secure Boot, bootloader, OTA image signing, RTC memory, BOOTLOADER_CUSTOM_RESERVE_RTC, partition table behavior, panic_abort, Guru Meditation, LoadStoreAlignment.
  - ROM/SoC specifics: esp_rom_spiflash_select_padsfunc, GPIO_OUTPUT_SET, GPIO_INPUT_GET, esp32c3/rom/gpio.h.
  - Build System internals: CMakeLists.txt content, idf_component_register, component names, per-component CMake behavior, Linux host (linux_compatible) build targets, unit-test build/coverage integration.
  - File paths: components/<lib>/* (esp_event, esp_rom, freertos, heap, spi_flash), components/bootloader/*.
- Keywords/phrases:
  - “bootloader,” “secure boot,” “OTA image/signing,” “RTC reserved memory,” “panic,” “Guru Meditation,” “freertos,” “heap regression,” “ODR violation,” “alignment,” “linker script,” “linux_compatible test app,” “component name,” “CMakeLists.txt changes.”
- Typical problems:
  - Crashes, panics, concurrency bugs, missing/removed symbols or APIs in ROM or components.
  - Behavior/semantics of runtime APIs (timekeeping, event handlers, Wi-Fi events).
  - Build logic feature requests/bugs within ESP-IDF CMake (component naming, hooks to run scripts after build, unit-test coverage on host).
  - CMake generator errors when invoking CMake directly for an ESP-IDF project (e.g., “CMake was unable to find Ninja” without any idf.py/install context).
- Ambiguous build issues:
  - If the complaint is about writing/using CMakeLists.txt, component selection, or project-side CMake logic, choose Core.
  - If an IRAM/DRAM overflow is about placement/linking of firmware sections, choose Core unless the complaint centers on idf_size output/analysis (then Tools).

Classify as IDF Tools when the issue is about host-side tooling, installation, environment setup, Python tooling, or ancillary developer tools:
- Technical indicators (commands, scripts, repos, variables):
  - idf.py and its subcommands (build, menuconfig, monitor), idf_tools.py, esp-idf-tools-setup-offline-*.exe, install.sh, install.ps1, export.sh.
  - idf_monitor tool (SerialMonitor.serial_write, RFC2217), idf_size, sbom_tool (esp-idf-sbom repo), clang-tidy-runner (run-clang-tidy.py, clang-tidy-diff.py).
  - Environment variables: IDF_TOOLS_PATH, IDF_PYTHON_ENV_PATH, proxy variables affecting downloads.
  - External tool deps: OpenOCD packaging, libusb on Linux, toolchain downloads, Python virtualenv management.
  - Repos/labels: esp-idf-monitor, esp-idf-sbom, clang-tidy-runner, “windows platform”, “tools”, “idf_monitor”, “idf_size”.
- Keywords/phrases:
  - “offline installer,” “Windows installer,” “install failed,” “download tools,” “export environment,” “Python env,” “virtual environment,” “menuconfig access violation,” “monitor colors,” “RFC2217,” “not writable IDF_TOOLS_PATH,” “libusb missing,” “OpenOCD install,” “proxy.”
- Typical problems:
  - Installation/setup failures, path/permission issues, missing dependencies on host OS.
  - idf.py behavior, menuconfig UI crashes, monitor hangs or protocol issues, idf_size reporting/analysis.
  - Requests to update offline packages or change idf.py behavior/policy.
- Ambiguous CMake/build errors:
  - If the error occurs while using idf.py or an installer/script sets up tools (e.g., Ninja not installed, Python env mismatch), choose Tools.
  - If the issue is about OS packaging, proxies, or dependency installation (libusb/OpenOCD), choose Tools.

Tie-breaker rules for mixed issues:
- Mentions of idf.py, idf_tools.py, install.sh/install.ps1/export.sh, Windows offline installer, or IDF_TOOLS_PATH/IDF_PYTHON_ENV_PATH override to IDF Tools.
- Mentions of bootloader/secure boot/RTC memory, ROM functions, FreeRTOS, or runtime crashes/panics override to IDF Core.
- Menuconfig:
  - Crash/access violation or UI/launch problems → Tools.
  - Misbehavior of a Kconfig option affecting firmware features/boot/runtime → Core.
- Memory/size:
  - Questions about idf_size tool output or size report correctness → Tools.
  - Linker section placement, IRAM/DRAM overflow due to code/ld script, runtime heap usage regressions → Core.

Quick examples to apply:
- “idf.py menuconfig Access violation” → Tools.
- “settimeofday sets local time not UTC” → Core.
- “esp-idf-tools-setup-offline-4.4.exe fails” → Tools.
- “BOOTLOADER_CUSTOM_RESERVE_RTC missing on ESP32C2” → Core.
- “Monitor hangs in SerialMonitor.serial_write with RFC2217” → Tools.
- “Unsubscribing from esp_event handler crashes” → Core.
- “CMake can’t find Ninja” while calling CMake directly with project CMakeLists → Core; if via idf.py or after running install scripts → Tools.

5. **BLE vs Classic Bluetooth Distinction:**:

1) Highest-priority technical indicators (APIs, headers, Kconfig, paths)
- Route to BLE if any of these appear:
  - Stacks/paths: components/bt/host/nimble/…, NimBLE, BLE Mesh (esp_ble_mesh_*), examples/bluetooth/bluedroid/ble/…, examples/bluetooth/esp_ble_mesh/…
  - APIs (BLE): esp_ble_gap_*, esp_ble_gatts_*, esp_ble_gattc_*, esp_blufi_*, ble_gap_*, ble_gattc_*, ble_gatts_*, ble_hs_*, ble_svc_*, blufi_init.c
  - Kconfig: CONFIG_BT_NIMBLE_*, CONFIG_BLE_MESH_*, CONFIG_BT_BLE_*, CONFIG_BT_CONTROLLER_MODE_BLE
  - BLE Mesh keywords: PB-ADV, PB-GATT, bearer, provision/provisioner, model, publish, LPN, friend
  - File/log strings: GATT/GATTS/GATTC, BLE_HS, NimBLE, BLE_MESH, advertising, scan, MTU, characteristic, descriptor
  - Examples: controller_vhci_ble_adv, gatt_server, gatt_client
- Route to Classic Bluetooth if any of these appear:
  - Profiles: SPP, A2DP, AVRCP, HFP/HSP, SCO/eSCO, PBAP, HID/HIDH, MAP, OPP
  - APIs (Classic): esp_spp_*, esp_a2d_*, esp_avrc_*, esp_hf_*/esp_ag_*, esp_bt_hid_* or esp_bt_hidh_*, esp_bt_gap_* (note: esp_bt_gap_* is Classic; BLE uses esp_ble_gap_*)
  - Kconfig: CONFIG_BT_CLASSIC_ENABLED, CONFIG_BT_SPP_ENABLED, CONFIG_A2DP_*, CONFIG_AVRCP_*, CONFIG_BT_HFP_*, CONFIG_BT_HID_*
  - Paths/examples: examples/bluetooth/bluedroid/classic/*, bt_avrc, a2dp_source, spp_acceptor
  - Classic-specific terms: RFCOMM, SDP/SDP record, COD/Class of Device (esp_bt_gap_set_cod), inquiry, page/page scan, BR/EDR, SCO
  - Log tags: BTA_, BTM_, A2D, AVRC, RFCOMM, SPP

2) Domain keywords and typical problem types
- BLE team typical issues:
  - Advertising/scanning visibility (e.g., not visible on iPhone/Android), whitelist, scan params
  - GATT client/server behavior, service/characteristic/descriptor, UUIDs 0x180X/0x2Axx, MTU issues
  - NimBLE compile/link/malloc/PSRAM issues, os_mempool.h, ble_gattc_disc_all_svcs
  - BLE Mesh build/runtime warnings/errors, provisioning, bearers, “No outbound bearer found”
  - BLUFI provisioning problems (esp_blufi_*)
  - BLE controller crashes or sleeps (btdm_sleep_check_duration) when context shows BLE/GATT/NimBLE
  - TX power for BLE, LE-specific VHCI usage (controller_vhci_ble_adv)
- Classic Bluetooth team typical issues:
  - Audio streaming/control (A2DP/AVRCP), cover art, audio glitches, reconnection behavior
  - SPP connect/acceptor throughput or stalls, RFCOMM/L2CAP over BR/EDR
  - Phone call audio/control (HFP/HSP), SCO link failures
  - HID keyboard/mouse/gamepad host/device (esp_bt_hidh_init)
  - GAP discovery/inquiry, device class (COD), pairing/pincode for Classic
  - Controller HCI data path issues when tied to ACL/SCO, Classic profiles, or esp_bt_gap_*

3) Error/log string cues
- BLE: “GATTS:”, “GATTC:”, “BLE_HS”, “NimBLE”, “BLE_MESH”, “No outbound bearer found”, “scan_evt timeout”, “adv”, “ATT”, “MTU”
- Classic: “AVRC”, “A2D”, “SPP”, “RFCOMM”, “BTM”, “BTA”, “SCO/eSCO”, “GAP: COD”
- Note: “GATT” implies BLE even if the word “Classic” appears elsewhere.

4) Chip-based decision aid
- If the chip is ESP32-C2/C3/C6/S3/H2 (no Classic Bluetooth support), any Bluetooth functionality issue must be routed to BLE.
- If ESP32 (original) and context is ambiguous, defer to API/profile clues above.

5) Ambiguous terms and tie-breakers
- GAP: Use the API prefix. esp_ble_gap_* => BLE. esp_bt_gap_* => Classic.
- L2CAP: If tied to ATT/MTU/GATT/UUIDs => BLE. If tied to RFCOMM/SDP/BR-EDR/SCO or Classic profiles => Classic.
- VHCI/HCI:
  - If example or commands explicitly say BLE/LE (e.g., “controller_vhci_ble_adv”, LE Set Advertising Parameters) => BLE.
  - If focused on ACL/SCO flow, Classic profiles, or Classic GAP => Classic.
- Coexistence with Wi‑Fi:
  - If BLE Mesh, GATT, advertising/scanning are involved => BLE.
  - If A2DP/SPP/AVRCP/HID/SCO are involved => Classic.
- “GATT” vs “Classic” conflict: Prefer BLE due to GATT being LE-only in ESP-IDF.
- Pairing issues:
  - If GATT server/client, characteristics, iPhone scan/advertise => BLE.
  - If PIN code, COD, SPP/A2DP/AVRCP/HID context => Classic.

6) File/build/link indicators
- BLE build failures typically reference: components/bt/host/nimble/…, ble_mesh/, blufi/, examples …/ble/…, symbols like ble_* or esp_ble_*.
- Classic build failures typically reference: a2dp_source, avrc_*, spp_*, hid/hidh, examples …/classic/…, symbols like esp_spp_*, esp_a2d_*, esp_avrc_*, esp_bt_gap_*.

7) Default rule when none of the above apply
- If any BLE-exclusive token appears (GATT, NimBLE, Mesh, BLUFI, advertising/scanning), classify as BLE.
- Else if any Classic-exclusive token appears (SPP, A2DP, AVRCP, HFP, HID/HIDH, SCO, COD, RFCOMM/SDP, esp_bt_gap_*), classify as Classic.
- If still uncertain and the chip is C2/C3/C6/S3/H2, classify as BLE; otherwise request more context, but tentatively classify based on example path or API prefixes present.

8) 
- Choose BLE when: Issue mentions "NimBLE", "nimble", "BLE", "GATT", "GAP", even if SPP is mentioned (NimBLE can emulate SPP over BLE)
- Choose Classic Bluetooth when: Issue mentions "Bluedroid", "Classic Bluetooth", "A2DP", "HFP", "SPP" WITHOUT NimBLE context
- Special case: "NimBLE SPP" or "Nimble SPP" → BLE team (SPP emulation over BLE)
Only choose from the provided team names (case-sensitive).
Provide clear technical reasoning for your choice based on the technical signals you identified."""

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
    """Prompt with component mappings and team-specific keywords."""
    
    # Build team-component mapping
    team_components_text = ""
    for team, components in team_to_components.items():
        components_list = ", ".join(components)
        team_components_text += f"• {team}: {components_list}\n"
    
    # Build team keywords mapping
    team_keywords_text = ""
    for team, keywords in team_keywords.items():
        if keywords:  # Only show teams that have keywords
            # Limit to first 20 keywords to keep prompt manageable
            keywords_display = keywords[:20]
            if len(keywords) > 20:
                keywords_display.append(f"... (+{len(keywords)-20} more)")
            keywords_list = ", ".join(keywords_display)
            team_keywords_text += f"• {team}: {keywords_list}\n"
    
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
        "CLASSIFICATION RULES:",
        "• Analyze issue summary and description for technical indicators",
        "• Match technical keywords, APIs, error codes, and function names to team expertise",
        "• Use the team technical keywords above to identify domain matches",
        "• Focus on the primary technical domain and root cause of the issue",
        "• When multiple teams could apply, choose the most specific technical match",
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
        
        # Find the original issue data for context
        issue_lookup = {issue.issue_key: issue for issue in issues}
        
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
