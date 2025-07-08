#!/usr/bin/env python
"""Run the Triton-generation pipeline on KernelBench problems.

Examples
--------
Run a single problem (level 1, id 3):
    python run_kernelbench.py --level 1 --problem 3

Run all problems of a level:
    python run_kernelbench.py --level 2

Environment
-----------
* GOOGLE_API_KEY must be set for the Gemini LLM.
* LOG_LEVEL (optional) controls log verbosity (DEBUG, INFO, ...).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import html  # For HTML entity unescaping
import requests
import urllib.parse
from pathlib import Path
from typing import List

from agents.pipeline_orchestrator import KernelPipelineAgent
from utils.kernelbench import build_op_spec, list_problems
from utils.logging_utils import configure_logging, get_logger

logger = get_logger("RunKB")


def _find_problem_file(level: int, problem_id: int) -> Path:
    """Locate the .py file whose numeric prefix equals `problem_id`. Handles files without zero-padding."""
    kb_root = Path(__file__).resolve().parent / "KernelBench/KernelBench" / f"level{level}"
    for pb_file in kb_root.glob("*.py"):
        try:
            pid = int(pb_file.name.split("_")[0])
        except ValueError:
            continue
        if pid == problem_id:
            return pb_file

    raise FileNotFoundError(
        f"Problem file not found for level {level}, id {problem_id}. "
        "Ensure the KernelBench dataset is present."
    )


def _run_single(level: int, problem_py: Path, problem_id: int) -> dict:
    op_spec = build_op_spec(problem_py, problem_id, level)
    pipeline = KernelPipelineAgent(web_search_tool_func=simple_web_search)
    return asyncio.run(pipeline.run(op_spec.model_dump()))


def simple_web_search(search_term: str) -> dict:
    """
    Advanced web search implementation following HuggingFace smolagents text_web_browser.py approach.
    
    Uses SerpAPI for high-quality results when available, with enhanced DuckDuckGo fallback.
    This provides better content parsing, multiple search strategies, and improved error handling.
    """
    import json
    import os
    
    try:
        # Try SerpAPI first (premium option for better results)
        serpapi_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPER_API_KEY")
        
        if serpapi_key:
            try:
                logger.info(f"Using SerpAPI for search: {search_term}")
                
                # SerpAPI Google Search (following smolagents approach)
                url = "https://serpapi.com/search"
                params = {
                    "q": search_term,
                    "engine": "google", 
                    "api_key": serpapi_key,
                    "num": 5,
                    "no_cache": "true",
                    "safe": "active"
                }
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                results = []
                organic_results = data.get("organic_results", [])
                
                for result in organic_results[:5]:
                    snippet = result.get("snippet", "")
                    title = result.get("title", "")
                    link = result.get("link", "")
                    
                    # Enhanced snippet with title context
                    enhanced_snippet = f"{title}. {snippet}" if title and snippet else (title or snippet or "No description available")
                    
                    results.append({
                        'snippet': enhanced_snippet,
                        'url': link,
                        'title': title,
                        'source': 'serpapi'
                    })
                
                # Add knowledge panel info if available
                knowledge_graph = data.get("knowledge_graph", {})
                if knowledge_graph.get("description"):
                    results.insert(0, {
                        'snippet': f"Knowledge Graph: {knowledge_graph['description']}",
                        'url': knowledge_graph.get("source", {}).get("link", ""),
                        'title': knowledge_graph.get("title", "Knowledge Graph"),
                        'source': 'serpapi_kg'
                    })
                
                if results:
                    logger.info(f"SerpAPI returned {len(results)} results for '{search_term}'")
                    return {'results': results}
                    
            except Exception as serpapi_error:
                logger.warning(f"SerpAPI search failed for '{search_term}': {serpapi_error}")
        
        # Enhanced DuckDuckGo fallback with multiple strategies
        logger.info(f"Using enhanced DuckDuckGo search for: {search_term}")
        
        encoded_query = urllib.parse.quote_plus(search_term)
        results = []
        
        # Strategy 1: DuckDuckGo Instant Answer API
        try:
            instant_api_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            }
            
            response = requests.get(instant_api_url, headers=headers, timeout=10)
            response.raise_for_status()
            instant_data = response.json()
            
            # Extract abstract
            if instant_data.get('Abstract'):
                results.append({
                    'snippet': instant_data['Abstract'],
                    'url': instant_data.get('AbstractURL', ''),
                    'title': instant_data.get('Heading', 'Abstract'),
                    'source': 'duckduckgo_instant'
                })
            
            # Extract definition if available
            if instant_data.get('Definition'):
                results.append({
                    'snippet': instant_data['Definition'],
                    'url': instant_data.get('DefinitionURL', ''),
                    'title': 'Definition',
                    'source': 'duckduckgo_definition'
                })
            
            # Extract related topics
            related_topics = instant_data.get('RelatedTopics', [])
            for topic in related_topics[:3]:  # Limit to avoid overwhelming
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'snippet': topic['Text'],
                        'url': topic.get('FirstURL', ''),
                        'title': 'Related Topic',
                        'source': 'duckduckgo_related'
                    })
                    
        except Exception as instant_error:
            logger.debug(f"DuckDuckGo Instant API failed: {instant_error}")
        
        # Strategy 2: If we need more results, try HTML scraping
        if len(results) < 3:
            try:
                html_search_url = f"https://duckduckgo.com/html/?q={encoded_query}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
                }
                
                response = requests.get(html_search_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse HTML for search results
                import re
                result_pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                matches = re.findall(result_pattern, response.text, flags=re.IGNORECASE | re.DOTALL)
                
                for url, title_html in matches[:5-len(results)]:
                    # Clean HTML from title
                    clean_title = re.sub(r'<[^>]+>', '', title_html)
                    clean_title = html.unescape(clean_title).strip()
                    
                    if clean_title and url:
                        results.append({
                            'snippet': clean_title,
                            'url': url,
                            'title': clean_title[:100] + "..." if len(clean_title) > 100 else clean_title,
                            'source': 'duckduckgo_html'
                        })
                        
            except Exception as html_error:
                logger.debug(f"DuckDuckGo HTML scraping failed: {html_error}")
        
        # Ensure we have at least one meaningful result
        if not results:
            # Provide helpful fallback suggestions based on search term
            fallback_suggestions = []
            
            if any(term in search_term.lower() for term in ['triton', 'gpu', 'cuda', 'kernel']):
                fallback_suggestions = [
                    "Check the official Triton documentation at https://triton-lang.org/",
                    "Search GitHub issues at https://github.com/openai/triton/issues",
                    "Look for examples in the Triton tutorials",
                    "Check Stack Overflow for Triton-related questions"
                ]
            elif any(term in search_term.lower() for term in ['compilation', 'error', 'debug']):
                fallback_suggestions = [
                    "Check official documentation for compilation guides",
                    "Search community forums and Q&A sites",
                    "Look for similar error reports on GitHub",
                    "Check project wikis and troubleshooting guides"
                ]
            else:
                fallback_suggestions = [
                    "Try searching on official documentation sites",
                    "Check relevant GitHub repositories and issues",
                    "Look for discussions on Stack Overflow",
                    "Search community forums and discussion boards"
                ]
            
            fallback_snippet = f"Search performed for: {search_term}. " + ". ".join(fallback_suggestions)
            
            results = [{
                'snippet': fallback_snippet,
                'url': f'https://duckduckgo.com/?q={encoded_query}',
                'title': 'Search Suggestions',
                'source': 'fallback'
            }]
        
        logger.info(f"Enhanced DuckDuckGo search returned {len(results)} results for '{search_term}'")
        return {'results': results}
        
    except Exception as e:
        logger.warning(f"All web search methods failed for '{search_term}': {e}")
        
        # Ultimate fallback with helpful guidance
        return {
            'results': [{
                'snippet': f"Web search temporarily unavailable for '{search_term}'. For technical issues like Triton compilation errors: 1) Check official documentation, 2) Search GitHub repositories and issues, 3) Look for Stack Overflow discussions, 4) Check community forums and wikis.",
                'url': '',
                'title': 'Search Unavailable (Guidance)',
                'source': 'error_fallback'
            }]
        }


def run(level: int, problem: int | None):
    if problem is not None:
        logger.info("Running single problem | level=%d id=%d", level, problem)
        pb_file = _find_problem_file(level, problem)
        result = _run_single(level, pb_file, problem)
        print(json.dumps(result, indent=2))
    else:
        logger.info("Running all problems in level %d", level)
        results: List[dict] = []
        for pb_file in list_problems(level):
            pid = int(pb_file.name.split("_")[0])
            logger.info("→ Problem %d", pid)
            res = _run_single(level, pb_file, pid)
            results.append({"problem": pb_file.name, "result": res})
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run the Triton pipeline on KernelBench problems"
    )
    parser.add_argument(
        "--level", type=int, required=True, choices=[1, 2, 3, 4], help="KernelBench difficulty level"
    )
    parser.add_argument(
        "--problem", type=int, help="Problem id within the level (omit to run all problems)"
    )

    args = parser.parse_args()
    run(level=args.level, problem=args.problem) 