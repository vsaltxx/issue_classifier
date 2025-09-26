#!/usr/bin/env python3
"""
This script chooses representatives from the train data.
Step 2 in a pipeline: choose representatives.
Representatives are those issues that are most representative of the entire dataset and will be used to train the classifier,
so they will be provided to the classifier as examples in each call.
"""

import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

NUM_REPRESENTATIVES = 1000
BATCH_SIZE = 20  # Select representatives in batches
NUM_BATCHES = 50   # 50 × 20 = 1000 total

# Multi-iteration filtering parameters
CHUNK_SIZE = 50  # Size of each chunk for all iterations
CANDIDATES_PER_CHUNK = 25  # Candidates to select from each chunk in each iteration

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_data(file_path, limit: int | None = None):
    with open(file_path, "r") as f:
        if limit is None:
            return [json.loads(line) for line in f]
        rows = []
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            rows.append(json.loads(line))
        return rows

def load_allowed_components() -> List[str]:
    with open("v_data/components.json", "r") as f:
        return json.load(f)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def filtering_schema() -> Dict[str, Any]:
    """JSON schema for filtering iterations: selecting candidates from chunks."""
    return {
        "type": "json_schema",
        "name": "select_candidates",
        "schema": {
            "type": "object",
            "properties": {
                "selected_issues": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Issue key (e.g., IDFGH-123)"
                    },
                    "description": f"Array of exactly {CANDIDATES_PER_CHUNK} selected candidate issue keys"
                }
            },
            "required": ["selected_issues"],
            "additionalProperties": False
        },
        "strict": True
    }


def final_selection_schema() -> Dict[str, Any]:
    """JSON schema for Stage 2: final representative selection."""
    return {
        "type": "json_schema",
        "name": "select_representatives",
        "schema": {
            "type": "object",
            "properties": {
                "selected_issues": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Issue key (e.g., IDFGH-123)"
                    },
                    "description": f"Array of exactly {BATCH_SIZE} selected issue keys"
                }
            },
            "required": ["selected_issues"],
            "additionalProperties": False
        },
        "strict": True
    }

def format_prompt(train_data, allowed_components, batch_num: int, total_batches: int, already_selected: List[str]):
    """Format prompt for selecting representatives in batches."""
    already_selected_text = ""
    if already_selected:
        already_selected_text = f"""
    Previously Selected Issues (DO NOT select these again):
    {', '.join(already_selected)}
    """
    
    prompt = f"""
    You are selecting representative ESP-IDF Jira issues from TRAIN (batch {batch_num}/{total_batches}).

    Objective
    - Select exactly {BATCH_SIZE} issues from this batch that best cover variety of components and topics.
    - This is batch {batch_num} of {total_batches}, aiming for {NUM_REPRESENTATIVES} total representatives.

    Rules
    - Prefer clear, canonical examples with strong signal between summary/description and the labeled component.
    - Ensure diversity across allowed components; avoid near-duplicates.
    - Do NOT select any issues that were already selected in previous batches.
    - Focus on different components/topics than previous selections for maximum diversity.

    Input
    - TRAIN: JSON array of objects with fields: issue_key, summary, description, components.
    {already_selected_text}

    Output Format
    - Return a JSON object with "selected_issues" array containing exactly {BATCH_SIZE} issue keys.
    - Each issue key should be exactly as it appears in TRAIN (e.g., "IDFGH-123").

    TRAIN:
    {json.dumps(train_data, ensure_ascii=False)}

    """
    return prompt


def format_filtering_prompt(chunk_data: List[Dict], allowed_components: List[str], chunk_num: int, total_chunks: int, iteration: int) -> str:
    """Format prompt for filtering iterations: selecting candidates from a chunk."""
    prompt = f"""
    You are selecting the most representative ESP-IDF Jira issues from a chunk (iteration {iteration}, chunk {chunk_num}/{total_chunks}).

    Objective
    - Select exactly {CANDIDATES_PER_CHUNK} issues from this chunk of {len(chunk_data)} issues.
    - Choose the most representative and diverse examples with the clearest signal.

    Rules
    - Prefer clear, canonical examples with strong signal between summary/description and the labeled component.
    - Ensure diversity across allowed components within this chunk.
    - Avoid near-duplicates or very similar issues.
    - Focus on issues that would be excellent training examples.
    - Prioritize issues with clear, unambiguous component assignments.

    Input
    - CHUNK: JSON array of {len(chunk_data)} objects with fields: key, summary, description, components.

    Output Format
    - Return a JSON object with "selected_issues" array containing exactly {CANDIDATES_PER_CHUNK} issue keys.
    - Each issue key should be exactly as it appears in CHUNK (e.g., "IDFGH-123").

    CHUNK:
    {json.dumps(chunk_data, ensure_ascii=False)}

    """
    return prompt


def filter_chunk(
    chunk_data: List[Dict], 
    allowed_components: List[str], 
    chunk_num: int, 
    total_chunks: int,
    iteration: int
) -> List[str]:
    """Filter a single chunk and select the best candidates."""
    log.info(f"Iteration {iteration}: Processing chunk {chunk_num}/{total_chunks} ({len(chunk_data)} issues)")
    
    prompt = format_filtering_prompt(chunk_data, allowed_components, chunk_num, total_chunks, iteration)
    
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        temperature=0,
        input=[
            {"role": "system", "content": "You are an expert at identifying the most representative examples from technical issue datasets."},
            {"role": "user", "content": prompt},
        ],
        text={"format": filtering_schema()},
        timeout=120,
    )
    
    # Parse the structured response
    result = json.loads(response.output_text)
    selected_issues = result.get("selected_issues", [])
    
    log.info(f"Iteration {iteration}, chunk {chunk_num}: selected {len(selected_issues)} candidates")
    return selected_issues


def select_batch_representatives(
    train_data: List[Dict], 
    allowed_components: List[str], 
    batch_num: int, 
    total_batches: int, 
    already_selected: List[str]
) -> List[str]:
    """Select representatives for a single batch using Responses API."""
    log.info(f"Selecting batch {batch_num}/{total_batches} ({BATCH_SIZE} representatives)")
    
    prompt = format_prompt(train_data, allowed_components, batch_num, total_batches, already_selected)
    
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        temperature=0,
        input=[
            {"role": "system", "content": "You are an expert at selecting diverse, representative examples from technical issue datasets."},
            {"role": "user", "content": prompt},
        ],
        text={"format": final_selection_schema()},
        timeout=120,
    )
    
    # Parse the structured response
    result = json.loads(response.output_text)
    selected_issues = result.get("selected_issues", [])
    
    log.info(f"Batch {batch_num} selected {len(selected_issues)} issues")
    return selected_issues


def run_filtering_iteration(data: List[Dict], allowed_components: List[str], iteration: int) -> List[Dict]:
    """Run one iteration of filtering: chunk data and select best candidates from each chunk."""
    log.info(f"Iteration {iteration}: Processing {len(data)} issues in chunks of {CHUNK_SIZE}")
    
    chunks = chunk_list(data, CHUNK_SIZE)
    log.info(f"Iteration {iteration}: Created {len(chunks)} chunks")
    
    all_candidates = []
    for i, chunk in enumerate(chunks, 1):
        candidates = filter_chunk(
            chunk_data=chunk,
            allowed_components=allowed_components,
            chunk_num=i,
            total_chunks=len(chunks),
            iteration=iteration
        )
        all_candidates.extend(candidates)
        log.info(f"Iteration {iteration}: Total candidates so far: {len(all_candidates)}")
    
    # Create filtered dataset for next iteration
    candidate_keys = set(all_candidates)
    filtered_data = [issue for issue in data if issue.get("key") in candidate_keys]
    
    log.info(f"Iteration {iteration} complete: {len(data)} → {len(filtered_data)} issues")
    return filtered_data


def main():
    log.info(f"Multi-iteration filtering: {NUM_REPRESENTATIVES} representatives from all available data")
    log.info(f"Strategy: chunks of {CHUNK_SIZE}, select {CANDIDATES_PER_CHUNK} from each chunk per iteration")
    
    # Load all training data (8,478 issues)
    log.info("Loading all training data...")
    current_data = load_data("v_data/done_issues.jsonl")
    
    def _truncate_desc(row):
        desc = row.get("description", "")
        row["description"] = (desc or "")[:100]  # Keep descriptions short to save tokens
        return row

    current_data = [_truncate_desc(dict(d)) for d in current_data]
    log.info(f"Loaded {len(current_data)} total issues")
    
    allowed_components = load_allowed_components()
    
    # Multi-iteration filtering
    iteration = 1
    while len(current_data) > NUM_REPRESENTATIVES * 1.5:  # Continue until we're close to target
        current_data = run_filtering_iteration(current_data, allowed_components, iteration)
        iteration += 1
        
        # Safety check to avoid infinite loops
        if iteration > 10:
            log.warning("Reached maximum iterations (10), proceeding to final selection")
            break
    
    log.info(f"Filtering complete after {iteration-1} iterations: {len(current_data)} candidates remain")
    
    # Final selection: Select exactly NUM_REPRESENTATIVES from remaining candidates
    if len(current_data) <= NUM_REPRESENTATIVES:
        # If we have fewer candidates than needed, use all of them
        final_selected = [issue.get("key") for issue in current_data if issue.get("key")]
        log.info(f"Using all {len(final_selected)} remaining candidates as representatives")
    else:
        # Use the existing batch selection logic for final selection
        log.info(f"Final selection: Selecting {NUM_REPRESENTATIVES} from {len(current_data)} candidates")
        all_selected = []
        
        for batch_num in range(1, NUM_BATCHES + 1):
            if len(all_selected) >= NUM_REPRESENTATIVES:
                break
                
            selected_batch = select_batch_representatives(
                train_data=current_data,
                allowed_components=allowed_components,
                batch_num=batch_num,
                total_batches=NUM_BATCHES,
                already_selected=all_selected
            )
            
            # Validate and add to results
            valid_selected = []
            for issue_key in selected_batch:
                if issue_key and issue_key not in all_selected and len(all_selected) < NUM_REPRESENTATIVES:
                    valid_selected.append(issue_key)
            
            all_selected.extend(valid_selected)
            log.info(f"Final batch {batch_num}: added {len(valid_selected)} issues (total: {len(all_selected)})")
        
        final_selected = all_selected
    
    # Write results
    log.info(f"Selection complete: {len(final_selected)} representatives selected from original {len(load_data('v_data/done_issues.jsonl'))} issues")
    
    # Write as CSV format (matching original format)
    content = ",".join(final_selected[:NUM_REPRESENTATIVES])  # Ensure exactly NUM_REPRESENTATIVES
    with open("v_data/representatives.txt", "w", encoding="utf-8") as f:
        f.write(content)
    
    log.info(f"Results written to v_data/representatives.txt ({len(final_selected[:NUM_REPRESENTATIVES])} representatives)")


if __name__ == "__main__":
    main()