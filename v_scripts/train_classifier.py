#!/usr/bin/env python3
import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================
TEST_PATH = "v_data/train_done_issues.jsonl"
COMPONENTS_PATH = "v_data/components.json"
OUT_PREDS_PATH = "predictions.json"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEST_LIMIT = 50
SHOW_DIAGNOSTICS = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assign-components")

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# =========================
# Data structures
# =========================
@dataclass
class IssueRow:
    issue_key: str
    summary: str
    description: str
    components: List[str]

# =========================
# IO helpers
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def read_components(path: str) -> List[str]:
    """Load component list"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_issue(raw: Dict[str, Any]) -> IssueRow:
    """Convert raw issue to IssueRow (no cleaning)"""
    key = (raw.get("issue_key") or raw.get("key") or "").strip()
    summary = raw.get("summary") or ""
    description = raw.get("description") or ""
    components = [str(c).strip() for c in (raw.get("components") or []) if str(c).strip()]
    return IssueRow(issue_key=key, summary=summary, description=description, components=components)

# =========================
# OpenAI schema
# =========================
SYSTEM_MSG = """You are an expert ESP-IDF engineer and component classifier with deep knowledge of the ESP-IDF framework.

Your task is to assign exactly one component to each GitHub issue based on technical analysis. 

Follow the analysis framework provided in the user prompt and choose only from the provided components (case-sensitive)."""

def response_schema(components: List[str]) -> Dict[str, Any]:
    """Create JSON schema for OpenAI response"""
    return {
        "name": "assign_components",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_key": {"type": "string"},
                            "components": {"type": "string", "enum": components},
                            "reasoning": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "technical_signals": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["issue_key", "components", "reasoning", "confidence", "technical_signals"],
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
# Evaluation
# =========================
def evaluate_accuracy(gold_by_key: Dict[str, List[str]], pred_by_key: Dict[str, str]) -> Dict[str, Any]:
    """Calculate accuracy metrics"""
    evaluated = correct = 0
    for k, gold in gold_by_key.items():
        if k in pred_by_key:
            evaluated += 1
            if pred_by_key[k] in gold:
                correct += 1
    return {
        "evaluated": evaluated, 
        "correct": correct, 
        "accuracy": (correct / evaluated) if evaluated else 0.0
    }

# =========================
# Main
# =========================
def main():
    """Main classification function"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load data
    components = read_components(COMPONENTS_PATH)
    raw_test = read_jsonl(TEST_PATH)
    log.info(f"Loaded: {len(components)} components, {len(raw_test)} test issues")

    # Normalize test data (no cleaning)
    test_rows = [normalize_issue(r) for r in raw_test[:TEST_LIMIT]]
    log.info(f"Processing {len(test_rows)} test issues")

    # Keep gold labels for evaluation
    gold_by_key = {r.issue_key: r.components for r in test_rows if r.components}
    log.info(f"Gold labels for {len(gold_by_key)} test issues")

    # Create simple component list for prompt
    component_list = "\n".join([f"- {comp}" for comp in components])

    # Predict for each test issue
    predictions = []
    for i, test in enumerate(test_rows, 1):
        log.info(f"Predicting {i}/{len(test_rows)}: {test.issue_key}")
        
        # Create test payload
        test_payload = {
            "issue_key": test.issue_key,
            "summary": test.summary,
            "description": test.description,
        }

        # Multi-step reasoning prompt
        prompt = f"""Analyze this ESP-IDF issue using the 4-step framework:

AVAILABLE COMPONENTS:
{component_list}

ISSUE TO CLASSIFY:
{json.dumps(test_payload, ensure_ascii=False, indent=2)}

CLASSIFICATION RULES:
- Driver + hardware → use specific driver_* component (not generic "driver") UNLESS it's a generic hardware issue
- Security/crypto/TLS → use mbedtls (not "Security")  
- HTTP server → use esp_http_server (not "HTTP Server")
- Standard C library functions → use newlib (not generic libraries)
- Compiler/linker/toolchain issues → use toolchain (not "tools")

COMPONENT PREFERENCE RULES:
- SD/MMC card issues → prefer "sdmmc" over "driver_sdmmc"
- Monitor/debugging tools → prefer "tools" over "idf_monitor" 
- Security/encryption/secure boot → prefer "Security" (capitalized)
- Flash encryption/secure boot → prefer "Security" over "bootloader"
- WiFi provisioning (wifi_prov_mgr, provisioning) → prefer "provisioning" over "Wi-Fi"
- JTAG/OpenOCD debugging → prefer "debugging and tracing" over "tools"
- Memory region overflow/IRAM → prefer "idf_size" over "toolchain"
- FAT filesystem (ff.c, f_mount) → prefer "storage" over "toolchain"
- Generic hardware issues (DMA, PSRAM, memory) → prefer "driver" over specific components
- Cross-component hardware issues → prefer "driver" over specific drivers
- Multi-channel RMT driver issues → prefer "driver" over "driver_rmt" 
- BLE advertising/GATT issues → prefer "BLE" over "bluedroid"
- BLE linking errors (esp_vhci_host_*) → prefer "BLE" over "toolchain"
- BLE mesh coexistence → prefer "BLE_Mesh" over "Coexistence"
- Windows installation issues → prefer "windows platform" over "tools"
- SDSPI specific issues → prefer "driver_sdspi" over "sdmmc"
- Monitor/IDF tools → prefer "tools" over "windows platform"
- PowerShell installation scripts → prefer "tools" over "windows platform"
- ROM functions (components/esp_rom/) → prefer "esp_rom" over "driver_gpio"
- ODR violations/C++ issues → prefer "cxx" over "mbedtls"
- USB device stack → prefer "usb_device" over "usb"
- USB serial JTAG console → prefer "usb_serial_jtag" over "driver_uart"
- GATT Classic Bluetooth → prefer "Bluetooth Classic" over "BLE"
- Modbus timer configuration → prefer "modbus" over "driver"
- RTC GPIO sleep → prefer "sleep and power management" over "driver"
- Clang-tidy tools → prefer "clang-tidy-runner" over "toolchain"

MULTI-STEP ANALYSIS FRAMEWORK:
Step 1: IDENTIFY technical signals
  - Function names (esp_wifi_*, gpio_*, nvs_*, i2c_*, uart_*, etc.)
  - Error codes (ESP_ERR_*, BLE_HS_*, ESP_FAIL, etc.)  
  - Hardware references (GPIO pins, I2C, SPI, UART, ADC, etc.)
  - File paths (components/*, drivers/*, examples/*)
  - Protocol names (HTTP, MQTT, TCP, UDP, WebSocket, etc.)

Step 2: MATCH signals to component responsibilities
  - Function prefixes → specific components (esp_wifi_* → Wi-Fi, gpio_* → driver_gpio)
  - Hardware terms → driver components (I2C/SDA/SCL → driver_I2C, UART → driver_uart)
  - Network protocols → networking components (TCP/socket → LWIP, HTTP → esp-protocols)
  - Error patterns → owning components (BLE_HS_* → nimble, NVS_* → nvs_flash)

Step 3: CHOOSE primary component
  - Select the component with PRIMARY responsibility for the issue
  - Prefer specific over general (driver_uart over driver, Wi-Fi over esp_netif)
  - Focus on the core functionality being reported

Step 4: PROVIDE confidence assessment
  - High: Clear function names, specific error codes, or exact component references
  - Medium: Hardware/protocol references with sufficient context
  - Low: General functionality descriptions without specific technical signals

CLASSIFICATION EXAMPLES:
- esp_wifi_connect() returns ESP_FAIL → Wi-Fi
- I2C device on GPIO 21/22 not responding → driver_I2C
- MQTT client connection timeout → esp-mqtt
- NVS namespace 'storage' not found → nvs_flash
- TCP socket bind() error → LWIP
- BLE_HS_ENOTCONN after pairing → nimble
- CMake build configuration error → Build System
- malloc() returns NULL, heap corruption → newlib
- stdatomic.h missing __atomic_test_and_set → newlib
- xtensa-esp32-elf-gcc compilation error → toolchain
- undefined reference to symbol, linker error → toolchain
- SD card mount failure, sdmmc_card_t → sdmmc
- idf.py monitor output issues → tools
- Flash encryption key generation → Security
- Secure boot configuration → Security
- wifi_prov_mgr scan failure → provisioning
- JTAG debug problem, OpenOCD → debugging and tracing
- region iram0_0_seg overflowed → idf_size
- ff.c compilation warning, FAT filesystem → storage
- DMA PSRAM hardware issue → driver (not esp_psram)
- BLE advertising with iPhone → BLE (not bluedroid)
- esp_vhci_host linking error → BLE (not toolchain)
- BLE mesh WiFi coexistence → BLE_Mesh (not Coexistence)
- Windows ESP-IDF installation → windows platform (not tools)
- SDSPI card initialization → driver_sdspi (not sdmmc)
- IDF monitor color issues → tools (not windows platform)
- install.ps1 PowerShell script → tools (not windows platform)
- components/esp_rom/include/ missing functions → esp_rom (not driver_gpio)
- ODR violation LoadStoreAlignment → cxx (not mbedtls)
- Web USB tinyusb stack → usb_device (not usb)
- CONFIG_ESP_CONSOLE_SECONDARY_NONE → usb_serial_jtag (not driver_uart)
- ESP_GATT_CONN_LMP_TIMEOUT classic → Bluetooth Classic (not BLE)
- CONFIG_FMB_TIMER_USE_ISR_DISPATCH_METHOD → modbus (not driver)
- RTC GPIO deep sleep wakeup → sleep and power management (not driver)
- idf_clang_tidy runner → clang-tidy-runner (not toolchain)
- RMT multi-channel 多通道输出支持 → driver (not driver_rmt)

Follow this systematic approach for accurate classification. Choose only from provided components (case-sensitive).
Provide your analysis following the structured reasoning format."""

        # Call OpenAI
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": response_schema(components)
            },
            timeout=60,
        )

        # Parse response
        obj = json.loads(resp.choices[0].message.content)
        pred = obj["predictions"][0]
        predictions.append({
            "issue_key": pred["issue_key"], 
            "components": pred["components"],
            "confidence": pred.get("confidence", "unknown"),
            "technical_signals": pred.get("technical_signals", []),
            "reasoning": pred.get("reasoning", "")
        })
        
        # Show confidence with emoji
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(pred.get("confidence", "unknown"), "⚪")
        signals_summary = ", ".join(pred.get("technical_signals", [])[:3])
        log.info(f"  → {pred['components']} {confidence_emoji} | Signals: {signals_summary}")

    # Save predictions
    with open(OUT_PREDS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(predictions)} predictions to {OUT_PREDS_PATH}")

    # Evaluate accuracy
    pred_by_key = {p["issue_key"]: p["components"] for p in predictions}
    metrics = evaluate_accuracy(gold_by_key, pred_by_key)

    # Confidence analysis
    from collections import Counter
    confidence_counts = Counter(p.get("confidence", "unknown") for p in predictions)
    log.info(f"Confidence distribution: {dict(confidence_counts)}")
    
    # Accuracy by confidence level
    confidence_accuracy = {}
    for conf in ["high", "medium", "low"]:
        conf_preds = {p["issue_key"]: p["components"] for p in predictions if p.get("confidence") == conf}
        conf_gold = {k: v for k, v in gold_by_key.items() if k in conf_preds}
        if conf_gold:
            conf_metrics = evaluate_accuracy(conf_gold, conf_preds)
            confidence_accuracy[conf] = conf_metrics["accuracy"]
    
    if confidence_accuracy:
        log.info(f"Accuracy by confidence: {confidence_accuracy}")

    # Show detailed mismatches if requested
    if SHOW_DIAGNOSTICS:
        miss_by_comp = Counter()
        mismatched_issues = []
        
        for k, gold in gold_by_key.items():
            if k in pred_by_key and pred_by_key[k] not in gold:
                # Find the original issue for context
                original_issue = None
                for test_row in test_rows:
                    if test_row.issue_key == k:
                        original_issue = test_row
                        break
                
                mismatched_issues.append({
                    'key': k,
                    'predicted': pred_by_key[k],
                    'expected': gold,
                    'summary': original_issue.summary[:100] if original_issue else "N/A"
                })
                
                for g in gold:
                    miss_by_comp[g] += 1
        
        log.info(f"Top missed components: {miss_by_comp.most_common(10)}")
        
        # Print detailed mismatches
        if mismatched_issues:
            log.info("\n\n\n=== MISMATCHED ISSUES (for debugging) ===")
            for issue in mismatched_issues:
                log.info(f"{issue['key']}:")
                log.info(f"  Predicted: \"{issue['predicted']}\"")
                log.info(f"  Expected:  {issue['expected']}")
                log.info(f"  Summary:   {issue['summary']}...")
                log.info("\n\n")

    # Enhanced metrics with confidence data
    enhanced_metrics = {
        **metrics,
        "confidence_distribution": dict(confidence_counts),
        "accuracy_by_confidence": confidence_accuracy
    }

    log.info(f"Evaluation: {enhanced_metrics}")
    print(json.dumps(enhanced_metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()