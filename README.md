# Online Appendix for "Criteria-based LLM Relevance Judgments"


### Approach Architecture
![4-Prompts Method Overview](./figures/Screenshot%202025-02-06%20at%2011.43.30%20PM.png)
*Figure 3: Overview of our criteria-based relevance judgment approach. The method evaluates each query-passage pair across four predefined criteria (Exactness, Topicality, Coverage, and Contextual Fit) before aggregating these scores into a final relevance judgment.*
## Prompts

## Code

Source Code (src/)

- File Structure:
  - src/
    - main.py: Entry point and argument processing
    - relevance_scoring.py: Core 4-prompts implementation
    - model_utils.py: Model configuration utilities
    - relevance_processors.py: Evaluation processing logic
  - data/
  - results/


Basic Usage
To evaluate passages using our criteria-based method:
```bash
python main.py \
  --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" \
  --queries_path "./data/llm4eval_query_2024.txt" \
  --docs_path "./data/llm4eval_document_2024.jsonl" \
  --result_file_path "./results/4prompts.txt" \
  --decomposed_relevance
```

For Together AI API usage:

```bash
# Set API key
export TOGETHER_API_KEY=your_api_key

# Run evaluation
python main.py [same arguments as above] -together
```


#### Language Models
- **google/flan-t5-large**
- **meta-llama/Meta-Llama-3-8B-Instruct** 
- **meta-llama/Llama-3.3-70B-Instruct-Turbo** (via Together AI)

## Experimental Results

### System Performance Comparison
![System Performance Scatter Plot](./figures/scatter_4prompts-llama3-8b_ndcg10.png)
*Figure 1: Comparison of NDCG@10 scores between manual judgments (y-axis) and our automatic relevance labels (x-axis). Red dots represent individual system performance, while blue dots show per-query/system scores. The strong diagonal alignment of red dots (systems) demonstrates our method's high effectiveness at ranking systems, achieving a 0.99 Spearman rank correlation with manual judgments.*

### Criteria Relationship Analysis
![Criteria Correlation Matrix](./figures/correlation_matrix_.png)
*Figure 2: Correlation matrix showing relationships between assessment criteria (T=Topicality, F=Contextual Fit, E=Exactness, C=Coverage) and relevance labels (J=Ground Truth, L=Predicted). Numbers 0-3 indicate grade levels. The matrix reveals strong correlations across criteria for non-relevant (0) and highly relevant (2-3) cases, with Topicality serving as a key differentiator for borderline relevance.*
