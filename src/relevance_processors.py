"""
Relevance judgment processors for multi-criteria evaluation approaches.
Handles:
- 4-prompts decomposed method (criteria-based)
"""

import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
from relevance_scoring import  get_relevance_score_decomposed_prompts
from prompts import *


class RelevanceProcessor:
    """Base class for processing relevance judgments."""
    
    def __init__(self, result_path: str):
        self.result_path = Path(result_path)
        self.setup_paths()
            
    def setup_paths(self):
        """
        Setup file paths for results and logging.
        Creates necessary directories for storing different types of outputs.
        """
        self.generation_path = self.result_path.parent / "generation_errors" / self.result_path.name
        self.logs_path = self.result_path.parent / "logs" / self.result_path.name.replace(".txt", ".jsonl")
        self.cuda_errors_path = self.result_path.parent / "cuda_errors" / self.result_path.name
        self.decomposed_path = self.result_path.parent / "decomposed_scores" / self.result_path.name.replace(".txt", ".jsonl")
        
        for path in [self.generation_path, self.logs_path, self.cuda_errors_path, self.decomposed_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
                
    def debug_print(self, qidx, docidx, pred_score, criteria_scores, prompt=None):
        """Debugging prints for first run."""
        if not hasattr(self, "first_run_complete"):
            self.first_run_complete = True
            if prompt:
                print("\n=== First Run Debug Info ===")
                print(f"Prompt:\n{prompt}")
            print(criteria_scores)
            print(f"Query ID: {qidx}")
            print(f"Doc ID: {docidx}")
            print(f"Prediction Score: {pred_score}")
            print("="*30)

def process_test_decomposed_prompts_only_qrel(test_qrel, docid_to_doc, qid_to_query, result_path: str, pipeline, system_message: str, max_query: Optional[int] = None, max_doc: Optional[int] = None) -> float:
    """Process relevance judgments using the 4-prompts decomposition method with query and document limits."""
    processor = RelevanceProcessor(result_path)
    decomposed_scores = {}
    total_llm_time = 0.0
    
    # Filter queries based on max_query
    unique_queries = test_qrel['qid'].unique()
    if max_query is not None:
        unique_queries = unique_queries[:max_query]
        test_qrel = test_qrel[test_qrel['qid'].isin(unique_queries)]
    
    # Group by query to apply max_doc limit
    if max_doc is not None:
        test_qrel = test_qrel.groupby('qid').head(max_doc).reset_index(drop=True)
    
    with open(processor.result_path, 'w') as result_file, \
         open(processor.generation_path, 'w') as generation_errors_file, \
         open(processor.cuda_errors_path, 'w') as cuda_errors_file:
        
        for eachline in tqdm(test_qrel.itertuples(index=True)):
            qidx = eachline.qid
            docidx = eachline.docid

            try:
                try:
                    pred_score, criteria_scores, llm_time = get_relevance_score_decomposed_prompts(
                        query=qid_to_query[qidx],
                        passage=docid_to_doc[docidx],
                        pipeline=pipeline,
                        log_file_path=processor.logs_path,
                        system_message=system_message,
                        qidx=qidx,
                        docidx=docidx
                    )
                except KeyError:
                    docidx = str(docidx)
                    pred_score, criteria_scores, llm_time = get_relevance_score_decomposed_prompts(
                        query=qid_to_query[qidx],
                        passage=docid_to_doc[docidx],
                        pipeline=pipeline,
                        log_file_path=processor.logs_path,
                        system_message=system_message,
                        qidx=qidx,
                        docidx=docidx
                    )
                
                total_llm_time += llm_time
                decomposed_scores[(qidx, docidx)] = criteria_scores
                
                processor.debug_print(qidx, docidx, pred_score, criteria_scores)
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    error_msg = f"CUDA out of memory error for docid {docidx}. Skipping this document."
                    print(error_msg)
                    cuda_errors_file.write(error_msg + '\n')
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                    continue
                else:
                    raise e

            try:
                score = int(pred_score)
                if 0 <= score <= 3:
                    result_file.write(f"{qidx} 0 {docidx} {score}\n")
                else:
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
            except (ValueError, TypeError):
                generation_errors_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                print(f"Invalid score for {qidx}, {docidx}: {pred_score}")
                result_file.write(f"{qidx} 0 {docidx} 0\n")

        # Save decomposed scores in JSONL format
        with open(processor.decomposed_path, 'w') as f:
            for k, v in decomposed_scores.items():
                json.dump({str(k): v}, f)
                f.write('\n')

    return total_llm_time


