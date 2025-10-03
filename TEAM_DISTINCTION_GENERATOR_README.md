# Team Distinction Rules Generator

## Overview

The `generate_team_distinction_rules.py` script is a specialized tool for improving ESP-IDF issue classification accuracy by generating AI-powered distinction rules between two specific teams. It analyzes historical issue patterns and uses LLM to extract distinguishing characteristics.

## Purpose

When your classification model struggles to distinguish between similar teams (e.g., "IDF Tools" vs "IDE", or "BLE" vs "Classic Bluetooth"), this script helps by:

1. **Analyzing real data**: Examines actual issues handled by each team
2. **Identifying patterns**: Finds technical indicators, keywords, and problem types unique to each team
3. **Generating rules**: Creates actionable classification instructions using AI analysis
4. **Improving accuracy**: Provides specific guidance to reduce misclassifications

## How It Works

```
Training Issues → Filter by Teams → LLM Analysis → Classification Rules
```

1. **Data Loading**: Loads team mappings from GitLab and training issues
2. **Team Selection**: Interactive selection of two teams to analyze
3. **Issue Filtering**: Finds issues that belong exclusively to each team
4. **Pattern Analysis**: LLM analyzes technical patterns in each team's issues
5. **Rule Generation**: Produces specific classification instructions
6. **Output**: Saves rules to file and displays them

## Usage

### Prerequisites

Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GITLAB_URL="https://your-gitlab-instance.com"
export GITLAB_TOKEN="your-gitlab-token"
export OPENAI_MODEL="gpt-5"  # optional
```

### Running the Script

```bash
cd /path/to/jira-classify
python v_scripts/generate_team_distinction_rules.py
```

### Interactive Process

1. **Team Selection**:
   ```
   Available teams (16):
    1. Application Utilities
    2. BLE
    3. Classic Bluetooth
    4. Chip Support
    5. IDF Core
    6. IDF Tools
    7. IDE
    ...
   
   Enter two team names to analyze:
   Team 1: IDF Tools
   Team 2: IDE
   ```

2. **Analysis**: The script automatically filters issues and generates rules

3. **Output**: Rules are saved to a file like `team_distinction_rules_idf_tools_vs_ide.txt`

## Example Output

```
# Team Distinction Rules: IDF Tools vs IDE

## Choose IDF Tools when:
- Issue mentions component manager, build system, or idf.py commands
- Build configuration or project setup problems
- Component resolution or dependency issues
- Build tool errors or configuration problems

## Choose IDE when:
- Issue mentions VS Code extension, Eclipse, or IDE-specific features
- Editor integration, syntax highlighting, or debugging UI problems
- IDE plugin installation or configuration issues
- Development environment setup specific to editors

## Key Technical Indicators:
- IDF Tools: "component manager", "idf.py", "build system", "cmake"
- IDE: "vscode", "extension", "intellisense", "debugging UI"
```

## Use Cases

### 1. Resolving Classification Conflicts
When your model consistently confuses two teams:
```bash
# Analyze the problematic teams
python v_scripts/generate_team_distinction_rules.py
# Select: IDF Tools vs Toolchains & Debuggers
```

### 2. Understanding Team Boundaries
To clarify which team handles what:
```bash
# Compare overlapping domains
python v_scripts/generate_team_distinction_rules.py
# Select: Chip Support vs Sleep and Power Management
```

### 3. Improving Model Accuracy
Add generated rules to your main classification prompt:
```python
# In team_classification.py, add to SYSTEM_MSG or classification rules:
"""
IDF Tools vs IDE Distinction:
- Choose IDF Tools for: build system, component manager, idf.py issues
- Choose IDE for: editor integration, VS Code extension, debugging UI issues
"""
```

## Requirements

- **Training Data**: `v_data/train_done_issues.jsonl` with labeled issues
- **GitLab Access**: For team/component mappings
- **OpenAI API**: For LLM analysis
- **Minimum Data**: At least 5 issues per team for meaningful analysis

## Output Files

Generated files follow the pattern:
```
team_distinction_rules_{team1}_vs_{team2}.txt
```

Examples:
- `team_distinction_rules_ble_vs_classic_bluetooth.txt`
- `team_distinction_rules_idf_tools_vs_ide.txt`
- `team_distinction_rules_chip_support_vs_sleep_and_power_management.txt`

## Integration

To use the generated rules in your main classifier:

1. **Copy relevant rules** from the generated file
2. **Add to system prompt** in `team_classification.py`
3. **Test the improvement** by running classification on test data
4. **Iterate** if needed by generating rules for other team pairs

## Troubleshooting

### Common Issues

1. **Empty API Response**:
   - Check OpenAI API key and model name
   - Verify internet connection
   - Check if prompt is too long (script limits to 15 examples per team)

2. **No Issues Found**:
   - Verify team names match exactly (case-sensitive)
   - Check if teams have enough training data
   - Ensure `v_data/train_done_issues.jsonl` exists and has component labels

3. **GitLab Connection Issues**:
   - Verify `GITLAB_URL` and `GITLAB_TOKEN`
   - Check network access to GitLab instance
   - Ensure token has read access to the repository

### Debug Mode

For detailed logging, the script automatically provides:
- Issue count per team
- Sample examples being analyzed
- API request/response information
- Generated rule content

## Best Practices

1. **Start with most problematic pairs**: Focus on teams that cause the most misclassifications
2. **Use sufficient data**: Ensure each team has at least 10-15 representative issues
3. **Review generated rules**: Manually verify that rules make technical sense
4. **Test incrementally**: Add rules for one team pair at a time and measure improvement
5. **Iterate**: Refine rules based on classification results

## Example Workflow

```bash
# 1. Identify problematic team pairs from classification results
# 2. Generate distinction rules
python v_scripts/generate_team_distinction_rules.py

# 3. Review and integrate rules into main classifier
# 4. Test improved classification
python v_scripts/team_classification.py

# 5. Repeat for other team pairs as needed
```

This tool is particularly valuable for fine-tuning classification accuracy in domains with overlapping responsibilities or similar technical vocabularies.
