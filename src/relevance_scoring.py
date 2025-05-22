"""
Relevance Scoring Module
-----------------------
Implements the 4-prompts method for decomposed relevance scoring of document passages.

The method evaluates each query-passage pair on four key criteria:
1. Exactness: How precisely the passage answers the query
2. Topicality: Subject matter alignment between query and passage 
3. Coverage: Amount of passage dedicated to query topic
4. Contextual Fit: Relevance of background/context provided

Each criterion is scored on a scale of 0-3:
0: Not relevant at all / No information provided
1: Marginally relevant / Partially addresses the criterion
2: Fairly relevant / Adequately addresses the criterion
3: Highly relevant / Fully satisfies the criterion
"""

from typing import Dict, Tuple, Optional
import json
import re
import torch
from model_utils import TogetherPipeline
import time

# Prompt Templates and System Messages
# ----------------------------------
SYSTEM_MESSAGE_DECOMPOSITION = """
Please assess how well the provided passage meets specific criteria in relation to the query. 
Use the following scoring scale (0-3) for evaluation:

0: Not relevant at all / No information provided.
1: Marginally relevant / Partially addresses the criterion.
2: Fairly relevant / Adequately addresses the criterion.
3: Highly relevant / Fully satisfies the criterion.
"""

CRITERIA_DEFINITIONS = {
    "Exactness": "How precisely does the passage answer the query.",
    "Topicality": "Is the passage about the same subject as the whole query (not only a single word of it).",
    "Coverage": "How much of the passage is dedicated to discussing the query and its related topics.",
    "Contextual Fit": "Does the passage provide relevant background or context."
}

def get_relevance_score_baseline(prompt: str, pipeline, system_message: str) -> Tuple[str, float]:
    """
    Get model response for a given prompt, handling both Together AI and standard pipelines.
    
    The function includes first-run logging to help with debugging and verification.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    
    # First-run logging for debugging
    if not hasattr(get_relevance_score_baseline, "called"):
        get_relevance_score_baseline.called = True
        print("Initial messages for verification:")
        print(messages)

    # Handle Together AI models
    if isinstance(pipeline, TogetherPipeline):
        if not hasattr(get_relevance_score_baseline, "output_from_together"):
            get_relevance_score_baseline.output_from_together = True
            print("Using Together AI model for inference")
        
        outputs = pipeline(messages)
        output = outputs[0]["generated_text"]
    
    # Handle standard pipeline models
    else:
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Process chat template if available
        if hasattr(pipeline.tokenizer, "apply_chat_template"):
            if hasattr(pipeline.tokenizer, 'chat_template') and pipeline.tokenizer.chat_template is not None:
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                if not hasattr(get_relevance_score_baseline, "warning"):
                    get_relevance_score_baseline.warning = True
                    print("Warning: No chat template available, using only the prompt (and not the system message).")
                prompt = f"{prompt}"
        else:
            prompt = f"{prompt}"


        strat_time = time.time()
        # Generate model output
        outputs = pipeline(
            prompt,
            max_new_tokens=100,
            eos_token_id=terminators,
            pad_token_id=128009,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        output = outputs[0]["generated_text"]

        llm_time = time.time() - strat_time

        # Return generated text without the prompt if chat template was used, otherwise return full text
        if hasattr(pipeline.tokenizer, 'chat_template') and pipeline.tokenizer.chat_template is not None:
            output =  outputs[0]["generated_text"][len(prompt):]
        else:
            output = outputs[0]["generated_text"]
    if not hasattr(get_relevance_score_baseline, "print_one_output"):
        get_relevance_score_baseline.print_one_output = True
        print(f"sample output: {output}")   
    return output, llm_time




def find_first_number(text: str) -> Optional[int]:
    """Extract the first valid relevance score (0-3) from text."""
    match = re.search(r'\b[0-3]\b', text)
    if match:
        return int(match.group())
    return 0

def get_criteria_score(query: str, passage: str, criteria: Tuple[str, str], 
                      pipeline, system_message_decomposition: str) ->Tuple[Optional[int], float]:
    """
    Score a specific criterion for the query-passage pair.
    
    Args:
        query: The search query
        passage: The document passage
        criteria: Tuple of (criterion name, criterion definition)
        pipeline: Model pipeline for scoring
        system_message_decomposition: System prompt for criterion evaluation
        
    Returns:
        Integer score 0-3 or None if invalid
    """
    criteria_name, criteria_definition = criteria
    criteria_prompt = (f"Please rate how well the given passage meets the {criteria_name} criterion "
                      f"in relation to the query. The output should be a single score (0-3) "
                      f"indicating {criteria_definition}")
    
    full_prompt = f"{criteria_prompt}\nQuery: {query}\nPassage: {passage}\nScore:"
    score, llm_time = get_relevance_score_baseline(full_prompt, pipeline, system_message_decomposition)
    # print(f"score: {score}")
    final_score = find_first_number(score)
    # print(final_score)
    return final_score, llm_time

def aggregate_scores(scores: Dict[str, int], query: str, passage: str,
                    pipeline, system_message: str) -> Tuple[Optional[int], float]:
    """
    Aggregate individual criteria scores into a final relevance score.
    
    Creates a prompt that shows the model all criteria scores and asks for
    a final relevance judgment that considers all aspects.
    """
    # Build prompt with all scores
    prompt = f"Query: {query}\nPassage: {passage}\n"
    for name, score in scores.items():
        prompt += f"\n{name}: {score}"
    
    # Add aggregation request
    prompt += ("\n\nPlease rate how the given passage is relevant to the query "
              "based on the given scores. The output must be only a score (0-3) "
              "that indicates how relevant they are.\nScore:")
    
    score, llm_time = get_relevance_score_baseline(prompt, pipeline, system_message)
    return find_first_number(score), llm_time


def get_relevance_score_decomposed_prompts(query: str, passage: str, pipeline, 
                                         log_file_path: str, system_message: str,
                                         qidx: str, docidx: str) -> Tuple[Optional[int], Dict[str, int], float]:
    """
    Main implementation of the 4-prompts decomposed scoring method.
    Returns (final_score, criteria_scores, total_llm_time).
    
    This function:
    1. Scores each criterion individually
    2. Aggregates scores into final relevance judgment
    3. Logs all scores and intermediate steps
    """
    decomposed_scores_dict = {}
    total_llm_time = 0.0
    
    # Score each criterion
    for criteria in CRITERIA_DEFINITIONS.items():
        score, llm_time = get_criteria_score(query, passage, criteria, pipeline, SYSTEM_MESSAGE_DECOMPOSITION)
        decomposed_scores_dict[criteria[0]] = score
        total_llm_time += llm_time

    # Aggregate scores
    final_score, agg_llm_time = aggregate_scores(decomposed_scores_dict, query, passage, pipeline, system_message)
    # total_llm_time += agg_llm_time

    # Log results for analysis
    scoring_log = {
        "qidx": qidx,
        "docidx": docidx,
        "query": query,
        "passage": passage,
        "decomposed_scores_dict": decomposed_scores_dict,
        "final_relevance_score": final_score,
        "total_llm_time": total_llm_time
    }

    with open(log_file_path, "a") as f:
        json.dump(scoring_log, f)
        f.write('\n')  # Ensure JSONL format

    return final_score, decomposed_scores_dict, total_llm_time