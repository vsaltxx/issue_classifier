#!/usr/bin/env python3
"""
Build keyword hints for component assignment using:
1) Statistical mining from training data
2) LLM expansion for synonyms/aliases
3) Merge with fixed keywords from component_keywords.json
"""

import os
import re
import json
import logging
from typing import List, Dict, Any
from collections import defaultdict, Counter
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================
TRAIN_PATH       = "v_data/train_done_issues.jsonl"
COMPONENTS_PATH  = "v_data/components.json"
OUT_HINTS_PATH   = "v_data/hints.json"

MODEL            = "gpt-4o-mini"
BATCH_SIZE       = 8    # components per LLM batch
SEEDS_PER_COMP   = 25   # top tokens to show LLM
MAX_KEYWORDS     = 30   # max keywords per component from LLM

MIN_TOKEN_LEN    = 2
MIN_COMP_SAMPLES = 3    # component needs >=3 training issues
MIN_TOKEN_COUNT  = 3    # token needs >=3 occurrences
TOP_TOKENS       = 40   # keep top 40 mined tokens per component

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build-hints")

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# =========================
# Text cleaning and tokenization
# =========================
STOP_WORDS = {
    # Basic English
    "a", "an", "the", "and", "or", "of", "to", "for", "with", "from", "on", "in", "at", "by", "as",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "can",
    "you", "your", "we", "our", "i", "me", "my", "they", "them", "this", "that", "it", "its",
    "but", "if", "when", "where", "how", "what", "which", "who", "why", "all", "any", "some", "no",
    
    # Technical generic
    "error", "issue", "bug", "fix", "code", "test", "build", "compile", "work", "use", "used",
    "function", "file", "example", "problem", "update", "version", "config", "set", "get",
    
    # Pronouns and demonstrative words
    "you", "your", "yours", "we", "our", "i", "me", "my", "they", "them", "their", "this", "that", "these", 
    "those", "it", "its", "he", "she", "him", "her", "his",
    
    # Auxiliary and linking verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "would", 
    "could", "should", "may", "might", "must", "shall", "will", "can", "cannot", "can't", "won't", "don't", 
    "doesn't", "didn't", "haven't", "hasn't", "hadn't", "wouldn't", "couldn't", "shouldn't", "mustn't",
    
    # Conjunctions and prepositions
    "but", "however", "also", "because", "since", "until", "while", "although", "though", "if", "unless", 
    "whether", "either", "neither", "both", "between", "among", "through", "during", "before", "after",
    
    # Interrogative and relative words
    "which", "what", "when", "where", "who", "whom", "whose", "whatever", "whenever", "wherever", "whoever",
    
    # Quantitative and qualitative words
    "much", "many", "little", "less", "least", "lot", "lots", "enough", "several", "various", "different",
    "same", "similar", "another", "other", "others", "else", "every", "each", "some", "any", "none", "one",
    "two", "three", "first", "second", "last", "next", "previous", "new", "old", "good", "bad", "best", "worst",
    
    # Logical and boolean values
    "true", "false", "null", "none", "yes", "no", "ok", "okay",
    
    # Polite words and greetings
    "please", "thanks", "thank", "hi", "hello", "regards", "sincerely", "best", "cheers",
    
    # Jira/GitHub specific words
    "issue", "bug", "feature", "problem", "request", "report", "fix", "update", "work", "example", "sample", 
    "code", "test", "testing", "reproduce", "reproduction", "minimal", "project", "build", "compile", 
    "compilation", "error", "warning", "failed", "failure", "pass", "passed", "success", "successful",
    "description", "summary", "title", "comment", "note", "notes", "edit", "edited", "update", "updated",
    
    # Markdown and markup
    "h1", "h2", "h3", "h4", "h5", "h6", "---", "###", "##", "#", "```", "`", "*", "**", "_", "__",
    
    # ESP-IDF and Espressif specific common words
    "esp", "idf", "espressif", "esp32", "esp32s2", "esp32s3", "esp32c3", "esp32c6", "esp32h2", "esp32p4",
    "chip", "soc", "revision", "version", "v1", "v2", "v3", "v4", "v5", "latest", "stable", "master",
    
    # General technical terms
    "api", "sdk", "library", "component", "module", "driver", "hal", "ll", "framework", "system", "config", 
    "configuration", "setup", "init", "initialize", "initialization", "start", "stop", "enable", "disable",
    "set", "get", "read", "write", "create", "delete", "open", "close", "connect", "disconnect", "include", 
    "header", "file", "function", "class", "struct", "enum", "macro", "define", "constant", "variable",
    "parameter", "argument", "return", "value", "type", "kind", "kinds", "kinds of", "kinds of errors",
    "kinds of errors", "kinds of errors", "kinds of errors", "kinds of errors", "kinds of errors", "kinds of errors",
    
    # Operating systems and tools
    "windows", "linux", "macos", "ubuntu", "debian", "cmake", "make", "gcc", "clang", "python", "bash", 
    "shell", "terminal", "console", "cmd", "powershell", "vscode", "ide", "editor",
    
    # Common technical words of low significance
    "device", "board", "kit", "development", "custom", "external", "internal", "default", "standard", 
    "basic", "advanced", "simple", "complex", "main", "primary", "secondary", "additional", "extra",
    "general", "specific", "common", "rare", "normal", "abnormal", "expected", "unexpected", "actual",
    "current", "previous", "following", "above", "below", "left", "right", "top", "bottom", "integer",
    "delete", "remove", "add", "append", "prepend", "insert", "replace", "modify", "update", "upgrade", 
    "compile", "build", "run", "test", "verify", "validate", "check", "ensure", "assert", "verify", "validate",
    "exit", "require", "requires", "required", "required by", "required by the", "required by the component",
    "program", "programs", "programmer", "programmers", "index", "automatically", "manually", "manual", "automatic",
    "here", "there", "everywhere", "anywhere", "union"

    
    # Units of measurement and numerals
    "ms", "us", "ns", "s", "sec", "min", "hour", "day", "week", "month", "year", "hz", "khz", "mhz", 
    "ghz", "v", "mv", "ma", "a", "w", "mw", "kb", "mb", "gb", "bit", "byte", "bytes",
    
    # Logging and debugging
    "log", "logs", "logging", "debug", "info", "warn", "warning", "error", "fatal", "trace", "verbose",
    "output", "input", "stdout", "stderr", "printf", "print", "show", "display", "dump",
}

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove URLs, code blocks, HTML
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\{code[^}]*\}.*?\{code\}', ' ', text, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[[^\]]*\|[^\]]*\]', ' ', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Extract and normalize tokens"""
    text = clean_text(text).lower()
    tokens = re.findall(r'[a-zA-Z0-9_+./-]+', text)
    
    result = []
    for token in tokens:
        if len(token) >= MIN_TOKEN_LEN and token not in STOP_WORDS:
            result.append(token)
            # Add variants
            if '_' in token:
                result.append(token.replace('_', ''))
    
    return list(dict.fromkeys(result))  # Remove duplicates, preserve order

# =========================
# Data loading
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def load_components(path: str) -> List[str]:
    """Load component list"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_component_keywords(path: str) -> Dict[str, List[str]]:
    """Load fixed component keywords"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning(f"Component keywords file not found: {path}")
        return {}

# =========================
# Token mining
# =========================
def mine_discriminative_tokens(train_data: List[Dict], components: List[str]) -> Dict[str, List[str]]:
    """Mine discriminative tokens for each component"""
    
    # Group training data by component
    comp_texts = defaultdict(list)
    for item in train_data:
        text = f"{item.get('summary', '')} {item.get('description', '')}"
        for comp in item.get('components', []):
            if comp in components:
                comp_texts[comp].append(text)
    
    # Filter components with enough samples
    comp_texts = {c: texts for c, texts in comp_texts.items() if len(texts) >= MIN_COMP_SAMPLES}
    log.info(f"Components with >={MIN_COMP_SAMPLES} samples: {len(comp_texts)}")
    
    # Count tokens globally and per component
    global_tokens = Counter()
    comp_tokens = defaultdict(Counter)
    
    for comp, texts in comp_texts.items():
        for text in texts:
            tokens = set(tokenize(text))  # Use set to count each token once per document
            for token in tokens:
                global_tokens[token] += 1
                comp_tokens[comp][token] += 1
    
    # Calculate discriminative scores and select top tokens
    result = {}
    for comp, token_counts in comp_tokens.items():
        scored_tokens = []
        for token, comp_count in token_counts.items():
            if comp_count >= MIN_TOKEN_COUNT:
                global_count = global_tokens[token]
                score = comp_count / max(global_count, 1)  # Discriminative score
                scored_tokens.append((token, score))
        
        # Sort by score and take top tokens
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        result[comp] = [token for token, _ in scored_tokens[:TOP_TOKENS]]
    
    return result

# =========================
# LLM expansion
# =========================
def expand_with_llm(client: OpenAI, component_tokens: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Expand tokens using LLM"""
    
    system_prompt = """You are an ESP-IDF expert. For each component, I'll give you seed tokens from training data.
Generate 10-20 additional SHORT keywords that users might mention in bug reports for that component.
Focus on: API names, abbreviations, related peripherals, common error terms.
Avoid: generic words like 'error', 'code', 'problem', 'issue'.
Return only lowercase keywords."""

    schema = {
        "type": "json_schema",
        "json_schema": {
        "name": "expand_keywords",
        "schema": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "component": {"type": "string"},
                            "keywords": {
                                "type": "array",
                                    "items": {"type": "string"},
                                "minItems": 5,
                                    "maxItems": MAX_KEYWORDS
                                }
                            },
                            "required": ["component", "keywords"]
                        }
                    }
                },
                "required": ["components"]
            }
        }
    }
    
    expanded = {}
    components = list(component_tokens.keys())
    
    # Process in batches
    for i in range(0, len(components), BATCH_SIZE):
        batch_comps = components[i:i+BATCH_SIZE]
        
        # Build prompt
        prompt_parts = ["Expand keywords for these ESP-IDF components:\n"]
        for comp in batch_comps:
            seeds = component_tokens[comp][:SEEDS_PER_COMP]
            prompt_parts.append(f"- {comp}: {seeds}")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=schema,
                temperature=0
            )
            
            data = json.loads(response.choices[0].message.content)
    for item in data.get("components", []):
        comp = item["component"]
                keywords = [kw.strip().lower() for kw in item.get("keywords", []) if kw.strip()]
                expanded[comp] = keywords
                
            log.info(f"Expanded batch {i//BATCH_SIZE + 1}/{(len(components)-1)//BATCH_SIZE + 1}")
            
        except Exception as e:
            log.warning(f"LLM expansion failed for batch {i//BATCH_SIZE + 1}: {e}")
    
    return expanded

# =========================
# Merge everything
# =========================
def build_final_hints(components: List[str], 
                      mined_tokens: Dict[str, List[str]], 
                      expanded_tokens: Dict[str, List[str]],
                      fixed_keywords: Dict[str, List[str]]) -> Dict[str, str]:
    """Build final keyword -> component mapping"""
    
    keyword_to_comp = {}
    
    # Add mined tokens
    for comp, tokens in mined_tokens.items():
        for token in tokens:
            keyword_to_comp[token] = comp
    
    # Add expanded tokens
    for comp, tokens in expanded_tokens.items():
        for token in tokens:
            keyword_to_comp[token] = comp
    
    # Add fixed keywords (these override everything else)
    for comp, keywords in fixed_keywords.items():
        for keyword in keywords:
            kw_lower = keyword.lower()
            keyword_to_comp[kw_lower] = comp
            
            # Add variants
            if '_' in kw_lower:
                keyword_to_comp[kw_lower.replace('_', '')] = comp
                keyword_to_comp[kw_lower.replace('_', ' ')] = comp
            
            if kw_lower.endswith('_t'):
                keyword_to_comp[kw_lower[:-2]] = comp
    
    return keyword_to_comp

# =========================
# Main
# =========================
def main():
    """Main function"""
    log.info("Building keyword hints...")
    
    # Load data
    components = load_components(COMPONENTS_PATH)
    train_data = load_jsonl(TRAIN_PATH)
    fixed_keywords = load_component_keywords("v_data/component_keywords.json")
    
    log.info(f"Loaded: {len(components)} components, {len(train_data)} training items")
    
    # Mine discriminative tokens
    mined_tokens = mine_discriminative_tokens(train_data, components)
    log.info(f"Mined tokens for {len(mined_tokens)} components")
    
    # Expand with LLM
    client = OpenAI(api_key=OPENAI_API_KEY)
    expanded_tokens = expand_with_llm(client, mined_tokens)
    log.info(f"LLM expanded tokens for {len(expanded_tokens)} components")
    
    # Build final hints
    hints = build_final_hints(components, mined_tokens, expanded_tokens, fixed_keywords)

    # Save
    os.makedirs(os.path.dirname(OUT_HINTS_PATH), exist_ok=True)
    with open(OUT_HINTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(hints, f, ensure_ascii=False, indent=2)
    
    log.info(f"Saved {len(hints)} hints to {OUT_HINTS_PATH}")
    print(f"Generated {len(hints)} keyword hints")

if __name__ == "__main__":
    main()