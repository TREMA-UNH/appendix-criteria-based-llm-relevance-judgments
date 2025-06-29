import torch
import argparse
import os
import logging
from typing import Optional
from data_processing import load_data_files, clean_files
from model_utils import get_model_baseline
from prompts import create_system_message
from relevance_processors import process_test_decomposed_prompts_only_qrel
import time

try:
    from umbrela_scoring import process_test_umbrella_prompt_only_qrel
except ImportError:
    process_test_umbrella_prompt_only_qrel = None


def setup_logging_and_device():
    logging.basicConfig(level=logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add these print statements to confirm
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Current device: {device}")
    
    # If on CUDA, you can also print additional details
    if torch.cuda.is_available():
        print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return logging.getLogger(__name__), device

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Process relevance judgments using multi-criteria evaluation.")
    parser.add_argument("--model_id", type=str, required=True, 
                      help="Model ID (e.g., meta-llama/Llama-3.3-70B-Instruct-Turbo)")
    parser.add_argument("--test_qrel_path", type=str, required=True,
                      help="Path to relevance judgments file")
    parser.add_argument("--queries_path", type=str, required=True,
                      help="Path to queries file")
    parser.add_argument("--docs_path", type=str, required=True,
                      help="Path to documents file")
    parser.add_argument("--result_file_path", type=str, required=True,
                      help="Output path for results")
    parser.add_argument("--score_order_in_prompt", type=str, default="3210",
                      help="Order of relevance scores in prompt")
    parser.add_argument("-decomposed_relavance", action="store_true",
                        help="Use 4-prompts method")
    parser.add_argument("-together", action="store_true",
                      help="Use together.ai API")
    
    ##FOR ANALYSIS

    parser.add_argument("--max_query", type=int, default=None,
                      help="Maximum number of queries to process")
    parser.add_argument("--max_doc", type=int, default=None,
                      help="Maximum number of documents per query to process")
    parser.add_argument("-umbrela_prompt", action="store_true",
                        help="running umbrela for comparison")              
    parser.add_argument("-time", action="store_true",
                                 help='computational time analysis')
    
    return parser.parse_args()
    
    

def main():
    logger, device = setup_logging_and_device()
    args = parse_arguments()

    docid_to_doc, qid_to_query, test_qrel = load_data_files(
        args.docs_path, args.queries_path, args.test_qrel_path
    )

    system_message = create_system_message(args.score_order_in_prompt)
    model = get_model_baseline(args.model_id, args.together)

    if args.decomposed_relavance:
        decomposed_llm_time = process_test_decomposed_prompts_only_qrel(
            test_qrel, docid_to_doc, qid_to_query,
            args.result_file_path, model, system_message,
            max_query=args.max_query, max_doc=args.max_doc
        )

    print(f"Multicriteria for {args.max_doc} doc per {args.max_query} query took {decomposed_llm_time:.2f} seconds") if args.time else None
    

    if args.umbrela_prompt:
        if process_test_umbrella_prompt_only_qrel is None:
            print("Warning: Umbrella prompt method skipped because umbrella_scoring.py is not available.")
        else:
            umbrella_llm_time = process_test_umbrella_prompt_only_qrel(
                test_qrel, docid_to_doc, qid_to_query,
                args.result_file_path, model, system_message,
                max_query=args.max_query, max_doc=args.max_doc
            )
            print(f"Umbrella prompt method LLM calls took {umbrella_llm_time:.2f} seconds") if args.time else None

            if args.decomposed_relavance:
                print(f"Time difference (Decomposed - Umbrella): {decomposed_llm_time - umbrella_llm_time:.2f} seconds") if args.time else None
                print(f"Decomposed is {decomposed_llm_time / umbrella_llm_time:.2f}x slower" if decomposed_llm_time > umbrella_llm_time else f"Umbrella is {umbrella_llm_time / decomposed_llm_time:.2f}x slower") if args.time else None

if __name__ == "__main__":
    main()