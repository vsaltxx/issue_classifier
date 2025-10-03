#!/usr/bin/env python3
"""
TEAM DISTINCTION RULES GENERATOR
===============================

This script generates AI classification rules to help distinguish between two specific ESP-IDF teams
by analyzing their historical issue patterns and using LLM to extract distinguishing characteristics.

WHAT IT DOES:
- Loads team mappings from GitLab YAML configuration
- Filters training issues that belong exclusively to each of two specified teams
- Analyzes technical patterns, keywords, and problem types for each team
- Uses OpenAI LLM to generate specific classification instructions
- Outputs actionable rules that can be added to classification prompts

HOW TO USE:
1. Ensure you have the required environment variables set:
   - OPENAI_API_KEY: Your OpenAI API key
   - GITLAB_URL: GitLab instance URL
   - GITLAB_TOKEN: GitLab access token
   - OPENAI_MODEL: Model name (optional, defaults to gpt-5)

2. Run the script:
   python v_scripts/generate_team_distinction_rules.py

3. The script will:
   - Show you all available teams from GitLab
   - Ask you to select two teams to analyze
   - Filter and analyze issues from both teams
   - Generate distinction rules using LLM
   - Save rules to a file and display them

EXAMPLE OUTPUT:
The script generates rules like:
"Choose Team A when: issue mentions X APIs, Y error codes, Z technical patterns
 Choose Team B when: issue mentions A APIs, B error codes, C technical patterns"

REQUIREMENTS:
- Training data in v_data/train_done_issues.jsonl
- Access to GitLab API for team/component mappings
- OpenAI API access
- At least 5 issues per team for meaningful analysis

USE CASES:
- Resolve classification conflicts between similar teams (e.g., IDF Tools vs IDE)
- Improve accuracy for teams with overlapping domains
- Generate team-specific classification guidance
- Understand technical boundaries between teams
"""

import os
import json
import logging
import urllib.parse
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import requests
import yaml

load_dotenv()

# =========================
# CONFIG
# =========================
TRAIN_PATH = "v_data/train_done_issues.jsonl"
TEST_PATH = "v_data/test_done_issues.jsonl"

# GitLab YAML source
GITLAB_URL = os.environ["GITLAB_URL"]
GITLAB_TOKEN = os.environ["GITLAB_TOKEN"]
PROJECT_PATH = "idf/idf-components-mapping"
FILE_PATH = "components.yaml"
REF = "main"

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Teams to analyze (will be set by user)
TEAM_1 = None
TEAM_2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("team-distinction")

# =========================
# Data loading utilities
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

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

def build_team_maps(doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    """Build component to team mapping"""
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
            if not name:
                continue
            name = str(name).strip()
            
            if name not in comp_to_team:
                comp_to_team[name] = team
            else:
                if comp_to_team[name] != team:
                    log.warning("Component %r appears under multiple teams: %r and %r (keeping first)",
                                name, comp_to_team[name], team)
    
    return team_names, comp_to_team

# =========================
# Issue filtering and analysis
# =========================
def filter_issues_by_teams(issues: List[Dict], comp_to_team: Dict[str, str], team1: str, team2: str) -> Tuple[List[Dict], List[Dict]]:
    """Filter issues that belong to team1 or team2 based on their components"""
    team1_issues = []
    team2_issues = []
    
    for issue in issues:
        components = issue.get("components", [])
        if not components:
            continue
            
        # Find which teams this issue belongs to
        issue_teams = set()
        for comp in components:
            if comp in comp_to_team:
                issue_teams.add(comp_to_team[comp])
        
        # Assign to team if it has components from that team
        if team1 in issue_teams and team2 not in issue_teams:
            team1_issues.append(issue)
        elif team2 in issue_teams and team1 not in issue_teams:
            team2_issues.append(issue)
        # Skip issues that belong to both teams or neither team
    
    return team1_issues, team2_issues

def prepare_issue_examples(issues: List[Dict], max_examples: int = 20) -> List[str]:
    """Prepare issue examples for LLM analysis"""
    examples = []
    
    for issue in issues[:max_examples]:
        summary = issue.get("summary", "").strip()
        description = issue.get("description", "").strip()
        components = ", ".join(issue.get("components", []))
        
        # Clean and truncate description
        if description:
            description = description[:500] + ("..." if len(description) > 500 else "")
        
        example = f"""Issue: {issue.get("issue_key", "N/A")}
Summary: {summary}
Components: {components}
Description: {description}"""
        
        examples.append(example)
    
    return examples

# =========================
# LLM analysis
# =========================
def generate_distinction_rules(client: OpenAI, team1: str, team2: str, team1_examples: List[str], team2_examples: List[str]) -> str:
    """Use LLM to analyze the differences and generate classification rules"""
    
    system_prompt = f"""You are an expert ESP-IDF technical analyst. Your task is to analyze issues from two different teams and generate precise classification instructions that would help distinguish between them.

You will be given examples of issues handled by:
- **{team1}**
- **{team2}**

Analyze the technical patterns, keywords, APIs, error types, and domains for each team. Then generate specific classification instructions that could be added to a prompt to help an AI model correctly distinguish between these two teams.

Focus on:
1. **Technical indicators** (APIs, function names, error codes, file paths)
2. **Domain-specific keywords** and terminology
3. **Types of problems** each team typically handles
4. **Clear decision rules** for ambiguous cases

Generate practical, actionable instructions that can be directly added to a classification prompt."""

    user_prompt = f"""Analyze these issue examples and generate classification instructions:

## {team1} Team Issues ({len(team1_examples)} examples):

{chr(10).join(f"{i+1}. {example}" for i, example in enumerate(team1_examples))}

## {team2} Team Issues ({len(team2_examples)} examples):

{chr(10).join(f"{i+1}. {example}" for i, example in enumerate(team2_examples))}

## Task:
Generate specific classification instructions that would help distinguish between {team1} and {team2} teams. Format as clear, actionable rules that can be added to a classification prompt."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    
    return response.choices[0].message.content

# =========================
# Main function
# =========================
def main():
    global TEAM_1, TEAM_2
    
    # Get team names from user
    print("Available teams will be loaded from GitLab...")
    
    # Load team mapping
    doc = fetch_components_yaml()
    team_names, comp_to_team = build_team_maps(doc)
    
    print(f"\nAvailable teams ({len(team_names)}):")
    for i, team in enumerate(sorted(team_names), 1):
        print(f"{i:2d}. {team}")
    
    # Get user input for teams
    print(f"\nEnter two team names to analyze:")
    TEAM_1 = input("Team 1: ").strip()
    TEAM_2 = input("Team 2: ").strip()
    
    if TEAM_1 not in team_names:
        print(f"Error: '{TEAM_1}' not found in available teams")
        return
    if TEAM_2 not in team_names:
        print(f"Error: '{TEAM_2}' not found in available teams")
        return
    
    print(f"\nAnalyzing distinction between '{TEAM_1}' and '{TEAM_2}'...")
    
    # Load training data
    train_issues = read_jsonl(TRAIN_PATH)
    log.info("Loaded %d training issues", len(train_issues))
    
    # Filter issues for the two teams
    team1_issues, team2_issues = filter_issues_by_teams(train_issues, comp_to_team, TEAM_1, TEAM_2)
    
    log.info("Found %d issues for '%s' team", len(team1_issues), TEAM_1)
    log.info("Found %d issues for '%s' team", len(team2_issues), TEAM_2)
    
    if len(team1_issues) < 5 or len(team2_issues) < 5:
        print(f"Warning: Not enough issues found for analysis")
        print(f"{TEAM_1}: {len(team1_issues)} issues")
        print(f"{TEAM_2}: {len(team2_issues)} issues")
        print("Need at least 5 issues per team for meaningful analysis")
        return
    
    # Prepare examples for LLM
    team1_examples = prepare_issue_examples(team1_issues, max_examples=15)
    team2_examples = prepare_issue_examples(team2_issues, max_examples=15)
    
    # Generate distinction rules using LLM
    client = OpenAI(api_key=OPENAI_API_KEY)
    log.info("Generating distinction rules using LLM...")
    
    distinction_rules = generate_distinction_rules(client, TEAM_1, TEAM_2, team1_examples, team2_examples)
    
    # Save results
    output_file = f"team_distinction_rules_{TEAM_1.replace(' ', '_').replace('&', 'and').lower()}_vs_{TEAM_2.replace(' ', '_').replace('&', 'and').lower()}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Team Distinction Rules: {TEAM_1} vs {TEAM_2}\n")
        f.write(f"# Generated from {len(team1_issues)} {TEAM_1} issues and {len(team2_issues)} {TEAM_2} issues\n\n")
        f.write(distinction_rules)
    
    print(f"\n" + "="*80)
    print(f"GENERATED DISTINCTION RULES: {TEAM_1} vs {TEAM_2}")
    print("="*80)
    print(distinction_rules)
    print("="*80)
    print(f"\nRules saved to: {output_file}")
    print(f"\nYou can now add these rules to your classification prompt to better distinguish between {TEAM_1} and {TEAM_2}.")

if __name__ == "__main__":
    main()
