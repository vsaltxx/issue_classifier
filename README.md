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
