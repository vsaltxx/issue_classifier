# ESP-IDF Jira Issue Classification

Automated classification system for ESP-IDF Jira issues using LLM-based approaches.

## Project Overview

This project aims to automatically classify ESP-IDF GitHub/Jira issues to the appropriate responsible teams based on technical content analysis. The system uses OpenAI's language model **gpt-5** to analyze issue descriptions and assign them to teams like Wi-Fi, BLE, Chip Support, Storage, etc.

## Architecture

- **Team-based classification**: `team_classification.py` - Classifies issues to responsible teams
- **Hint generation**: `build_team_hints.py` - Automatically extracts team-specific technical patterns
- **Data processing**: Various experimenting scripts for data preparation and analysis

## Experiments Log

### 2025-10-02-1 — Team Classification (teams)
- **Model:** `gpt-5`
- **Train/Test:** `19 test`
- **Prompt features:** `3-step analysis framework, 341 auto-extracted keywords, detailed team descriptions, structured output`
- **Result:** `accuracy = 0.895`
- **Observations:** 
  - Good performance on clear technical signals
  - Edge cases: confusion between symptoms vs root domain (wakeup→power mgmt vs GPIO driver)
  - Model focuses on error type rather than technical domain



### 2025-10-02-2 — Enhanced Keyword Extraction (teams)
- **Model:** `gpt-5`
- **Train/Test:** `100 examples per team, all teams processed at once`
- **Prompt features:** `single LLM call for all teams, 4-category pattern extraction (APIs/errors/structures/terms), 100 examples per team`
- **Result:** `559 keywords across 16 teams`
- **Observations:** 
Loaded: 16 teams, 132 component mappings
 Loaded: 5961 training items
 Prepared examples for 16 teams
   IDF Core: 100 examples
   Storage: 100 examples
   IDF Tools: 100 examples
   Chip Support: 100 examples
   Wi-Fi: 100 examples
   Application Utilities: 100 examples
   Networking and Protocols: 100 examples
   Toolchains & Debuggers: 100 examples
   BLE: 100 examples
   USB: 100 examples
   Classic Bluetooth: 100 examples
   Sleep and Power Management: 100 examples
   802.15.4: 95 examples
   Other: 96 examples
   IDE: 45 examples
   Security: 1 examples
--
 38 patterns for idf core team
 30 patterns for storage team
 27 patterns for idf tools team
 24 patterns for chip support team
 34 patterns for wi-fi team
 21 patterns for application utilities team
 27 patterns for networking and protocols team
 24 patterns for toolchains & debuggers team
 30 patterns for ble team
 27 patterns for usb team
 19 patterns for classic bluetooth team
 19 patterns for sleep and power management team
 29 patterns for 802.15.4 team
 19 patterns for other team
 20 patterns for ide team
 7 patterns for security team
2025-10-02 13:21:51,302 INFO Successfully processed all teams in single call
2025-10-02 13:21:51,302 INFO LLM extracted patterns for 16 teams
2025-10-02 13:21:51,303 INFO Saved 559 team hints to v_data/team_hints.json

### 2025-10-02-3 — Optimized Keywords & Build System Rules (teams)
- **Model:** `gpt-5`
- **Train/Test:** `100 samples`
- **Prompt features:** `15 unique keywords per team (240 total), build system disambiguation rules (IDF Core vs IDF Tools), enhanced classification examples, batch parallel processing (5 issues per API call)`
- **Result:** `accuracy = 0.874`
- **Observations:** 
  - Specified instruction how to classify build system from IDF Tools and IDF Core
  - Reduced the number of keywords to 15 per team (from 559 to 240 total)
  - Added clear routing rules for component usage vs build system internals
  - Implemented batch parallel processing for 5x faster API calls (100 issues → 20 API calls)

### 2025-10-02-4 — Enhanced Rules Based on Mismatch Analysis (teams)
- **Model:** `gpt-5`
- **Train/Test:** `100 samples`
- **Prompt features:** `mismatch-driven rule enhancement, strengthened component-first priority, specific component mappings (usb_serial_jtag→Chip Support, tools→IDF Tools), critical build system disambiguation, driver vs power management rules, USB vs serial disambiguation, batch processing (20 issues per API call)`
- **Result:** `accuracy = 0.874`
- **Observations:** 
  - BATCH_SIZE = 20 # Process N issues at once --> a lot faster
  - Analyzed 12 mismatches from previous run (87.4% accuracy) to identify root causes
  - Added 5 new rule categories to address: Build System confusion (5 issues), Component mapping gaps (3 issues), Content over component priority (2 issues), Driver/power confusion (1 issue), Language standard placement (1 issue)
  - Enhanced component-first rule with explicit examples and stronger prioritization

### 2025-10-02-5 — Absolute Component-First Rules (teams)
- **Model:** `gpt-5`
- **Train/Test:** `100 samples`
- **Prompt features:** `absolute component-first rule overriding all description analysis, critical component mappings with exact matches, mandatory driver component rules (all driver_* → Chip Support), strict BLE vs Classic Bluetooth disambiguation, build system override rule, comprehensive component categories`
- **Result:** `accuracy = 0.926`
- **Observations:** 
  - Implemented absolute component field priority over description content
  - Fixed 5 of 12 previous mismatches, reducing errors from 12 to 7 issues
  - Significant improvement: 87.4% → 92.6% (+5.2 percentage points)
  - Remaining 7 mismatches are edge cases requiring deeper analysis

### 2025-10-03-1 — Detailed Team Distinction Rules Experiment (teams)
- **Model:** `gpt-5`
- **Train/Test:** `100 samples`
- **Prompt features:** `generated extremely detailed distinguishing rules between specific team pairs (BLE vs Classic Bluetooth, IDF Tools vs IDE, Sleep vs Chip Support, etc.), comprehensive technical indicators, extensive API lists, file path patterns, error code mappings`
- **Result:** `accuracy decreased significantly`
- **Observations:** 
  - **Failed experiment**: Generated detailed distinction rules for team pairs using comprehensive technical analysis
  - Model became overwhelmed by the excessive detail and rule complexity
  - Accuracy dropped compared to simpler, more focused approaches
  - **Key learning**: More detailed rules ≠ better performance; concise, high-signal rules work better
  - Led to pivot toward few-shot contrastive examples instead of verbose rule sets

### 2025-10-03-2 — Concise System + Few-Shot Examples (teams)
- **Model:** `gpt-5`
- **Train/Test:** `100 samples`
- **Prompt features:** `concise system message (≤300 tokens), contrastive few-shot examples for tough boundaries, removed "Other" team option, optional reasoning field, mdns component dependency fix`
- **Result:** `accuracy = 0.937`
- **Observations:** 
  - Reduced system message from ~2000+ to ~200 tokens with core API prefix rules only
  - Added 15 contrastive examples for key team boundaries (Tools vs IDE, Core vs Tools, BLE vs Classic, etc.)
  - Fixed IDFGH-11204 mdns dependency issue with specific example
  - Removed "Other" team forcing more precise classification
  - **Major improvement**: 86.3% → 93.7% (+7.4 percentage points) after mismatch analysis
  - Only 6 remaining mismatches, mostly edge cases with "Other" predictions

### 2025-10-03-3 — Priority-Based Classification with Component Matching (teams)
- **Model:** `gpt-5`
- **Train/Test:** `500 samples`
- **Prompt features:** `priority-based decision framework (P1-P5), exact component name matching with word boundaries, anti-"Other" guardrails, structured API prefix hierarchy, path/config/error pattern matching, JSON-only output format`
- **Result:** `accuracy = 0.856`
- **Observations:** 
  - **Regression**: 93.7% → 85.6% (-8.1 percentage points) despite more sophisticated approach
  - Implemented P0 component name matching with constraints for weak/umbrella terms
  - Added strict priority hierarchy: Component names → API prefixes → Paths → Config → Errors → Context
  - Expanded test set to 500 samples revealing more edge cases (69 mismatches)
  - **Key insight**: More complex priority systems can hurt performance vs simpler approaches
  - Many mismatches show boundary confusion: esp_http_* (App Utilities vs Networking), USB vs Chip Support, build system ownership

### 2025-10-03-4 — Hard Guardrails + Explicit Overrides (teams)
- **Model:** `gpt-5`
- **Train/Test:** `500 samples`
- **Prompt features:** `hard anti-"Other" guardrail with ESP-IDF token list, 8 explicit override rules for major confusion pairs, simplified classification process (overrides → API prefixes → paths → context), targeted fixes for App Utils vs Networking, IDF Tools vs IDE, Core vs Tools boundaries`
- **Result:** `accuracy = 0.860`
- **Observations:** 
  - **Marginal improvement**: 85.6% → 86.0% (+0.4 percentage points)
  - Reduced mismatches from 69 to 67 issues
  - Hard guardrail prevents false "Other" classifications when ESP-IDF tokens present
  - Explicit overrides address systematic confusion pairs but limited impact
  - **Key learning**: Targeted fixes help but major accuracy gains require different approaches
  - Remaining 67 mismatches suggest need for more fundamental classification strategy changes



## Log template:

### YYYY-MM-DD — <Approach name> (<components|teams>)
- **Model:** `gpt-5`
- **Train/Test:** `19 train`
- **Prompt features:** `<bulleted key features>`
- **Result:** `accuracy = 0.XXX`
- **Observations:** `<1–3 bullets>`


---

## Usage

### Generate Team Hints
```bash
python v_scripts/build_team_hints.py
```

### Run Team Classification
```bash
python v_scripts/team_classification.py
```

### Run Component Classification -> needs to be updated!!!
```bash
python v_scripts/train_classifier.py
```

## Team Structure

Teams are loaded from GitLab YAML configuration:
- **Chip Support**: Hardware drivers (GPIO, I2C, SPI, UART, etc.)
- **Wi-Fi**: WiFi connectivity and related APIs
- **BLE**: Bluetooth Low Energy functionality
- **Classic Bluetooth**: Classic Bluetooth protocols
- **Networking and Protocols**: TCP/IP, HTTP, MQTT, etc.
- **Application Utilities**: HTTP server, TLS, OTA, provisioning
- **Storage**: NVS, filesystems, SD cards
- **IDF Core**: FreeRTOS, system core, bootloader
- **IDF Tools**: Build system, monitoring tools
- **Toolchains & Debuggers**: GCC, GDB, debugging tools
- **USB**: USB host/device functionality
- **802.15.4**: Thread, Zigbee, Matter protocols
- **Sleep and Power Management**: Power management features
- **Security**: Cryptographic functions, secure boot
- **IDE**: Development environment tools
- **Other**: Testing, CI, documentation, hardware issues
