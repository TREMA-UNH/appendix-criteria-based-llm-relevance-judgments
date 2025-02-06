
 ## Constants

MAX_LENGTH = 8000

# system_message = """You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

# 3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
# 2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
# 1 = Related: The passage seems related to the query but does not answer it.
# 0 = Irrelevant: The passage has nothing to do with the query

# Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""




def get_criteria_prompt(criteria: str, criteria_definition: str, query: str, passage: str) -> str:
    return f"""Please rate how well the given passage meets the {criteria} criterion in relation to the query. The output should be a single score (0-3) indicating {criteria_definition}.

    Query: {query}
    Passage: {passage}
    Score:"""

def get_final_relevance_prompt(query: str, passage: str, scores: dict) -> str:
    scores_text = "\n".join([f"{k}: {v}" for k,v in scores.items()])
    return f"""Please rate how the given passage is relevant to the query based on the given scores. The output must be only a score (0-3) that indicates how relevant they are.

    Query: {query}
    Passage: {passage}
    {scores_text}
    Score:"""

def parse_order(order_str):
    try:
        order = [int(char) for char in order_str]
        if sorted(order) != list(range(len(order))):
            raise ValueError("Invalid order string")
        return order
    except ValueError:
        return "Invalid order string"
    
   
# Function to create prompts based on different orders
def create_system_message(order_str):
    order = parse_order(order_str)
    descriptions = {
        0: "0 = Irrelevant: The passage has nothing to do with the query.",
        1: "1 = Related: The passage seems related to the query but does not answer it.",
        2: "2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.",
        3: "3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer."
    }
    ordered_descriptions = "\n    ".join(descriptions[i] for i in order)
    prompt = f"""You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n
    {ordered_descriptions}
    Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""
    print(prompt)
    return prompt

def create_system_message_for_rel(order_str="32"):
    order = parse_order(order_str)
    descriptions = {
        
        2: "2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.",
        3: "3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer."
    }
    ordered_descriptions = "\n    ".join(descriptions[i] for i in order)
    prompt = f"""You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 2 or 3 with the following meanings:\n
    {ordered_descriptions}"""
    # Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""
    print(prompt)
    return prompt

def create_system_message_for_non_rel(order_str):
    order = parse_order(order_str="10")
    descriptions = {
        0: "0 = Irrelevant: The passage has nothing to do with the query.",
        1: "1 = Related: The passage seems related to the query but does not answer it.",

    }
    ordered_descriptions = "\n    ".join(descriptions[i] for i in order)
    prompt = f"""You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 or 1 with the following meanings:\n
    {ordered_descriptions}"""
    # Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""
    print(prompt)
    return prompt

def get_prompt(query, passage,pipeline):

    prompt = f"""Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.

    Query: {query}
    Passage: {passage}

    Score:"""
    return prompt
    # return truncate_prompt_based_on_passage(prompt, pipeline, MAX_LENGTH)

   





def truncate_prompt_based_on_passage(prompt:str,pipeline, max_length: int) -> str:
    # Truncate passage part of the prompt
    """Truncate passage in the prompt if it exceeds the maximum token length."""
    tokens = pipeline.tokenizer.tokenize(prompt)
    if len(tokens) <= max_length:
        return prompt

    passage_start_index = prompt.find("Passage:") + len("Passage:")
    passage_end_index = prompt.find("Score:")
    truncated_passage = prompt[passage_start_index:passage_end_index]

    passage_tokens = pipeline.tokenizer.tokenize(truncated_passage)
    prompt_tokens = pipeline.tokenizer.tokenize(prompt[:passage_start_index]) + pipeline.tokenizer.tokenize(prompt[passage_end_index:])
    available_length = max_length - len(prompt_tokens)

    truncated_passage_tokens = passage_tokens[:available_length]
    # print("here")
    print(truncated_passage_tokens)
    truncated_passage = pipeline.tokenizer.decode(truncated_passage_tokens[1])
    # print(f"{prompt[:passage_start_index]} {truncated_passage} {prompt[passage_end_index:]}")
    return f"{prompt[:passage_start_index]} {truncated_passage} {prompt[passage_end_index:]}"


