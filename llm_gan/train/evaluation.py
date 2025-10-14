"""Judge evaluation functions for story assessment."""

import random
from typing import List, Tuple, Dict, Any

from llm_gan.prompts import llm_generator_discriminator_prompt
from llm_gan.utils.parse import parse_boxed
from .inference import simple_generate


def assess_judge(titles: List[str], genres: List[str], stories_human: List[str], 
                stories_ai: List[str], judge_model, tokenizer, max_tokens: int = 512) -> List[bool]:
    """Assess judge performance on human vs AI stories."""
    targets = [random.randint(0, 1) for _ in range(len(titles))]
    
    prompts = []
    for title, genre, human_story, ai_story, target in zip(titles, genres, stories_human, stories_ai, targets):
        story_order = (human_story, ai_story) if target == 0 else (ai_story, human_story)
        prompt = llm_generator_discriminator_prompt(title, genre, *story_order)
        prompts.append(prompt)
    
    judge_outputs = simple_generate(
        model=judge_model, 
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_tokens,
        batch_size=len(prompts)
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        # Extract content after assistant header to get just the response
        if "assistant<|end_header_id|>" in output:
            response = output.split("assistant<|end_header_id|>")[-1].strip()
        elif "<|start_header_id|>assistant<|end_header_id|>" in output:
            response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = output.strip()
        
        # Parse boxed answer from the response only
        answer = parse_boxed(response)
        
        if answer == "1":
            parsed_outputs.append(0)
        elif answer == "2":
            parsed_outputs.append(1)
        else:
            parsed_outputs.append(-1)
    
    correct = [parsed == target for parsed, target in zip(parsed_outputs, targets)]
    return correct


def assess_judge_with_outputs(titles: List[str], genres: List[str], stories_human: List[str], 
                            stories_ai: List[str], judge_model, tokenizer, 
                            max_tokens: int = 512) -> Tuple[List[bool], Dict[str, Any]]:
    """Assess judge performance and return detailed outputs."""
    targets = [random.randint(0, 1) for _ in range(len(titles))]
    
    prompts = []
    for title, genre, human_story, ai_story, target in zip(titles, genres, stories_human, stories_ai, targets):
        story_order = (human_story, ai_story) if target == 0 else (ai_story, human_story)
        prompt = llm_generator_discriminator_prompt(title, genre, *story_order)
        prompts.append(prompt)
    
    judge_outputs = simple_generate(
        model=judge_model, 
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_tokens,
        batch_size=len(prompts)
    )
    
    # Extract just the judge responses (not the full prompt+response) and parse them
    judge_responses = []
    parsed_outputs = []
    
    for output in judge_outputs:
        # Extract content after assistant header to get just the response
        if "assistant<|end_header_id|>" in output:
            response = output.split("assistant<|end_header_id|>")[-1].strip()
        elif "<|start_header_id|>assistant<|end_header_id|>" in output:
            response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = output.strip()
        
        judge_responses.append(response)
        
        # Parse boxed answer from the response only
        answer = parse_boxed(response)
        
        if answer == "1":
            parsed_outputs.append(0)
        elif answer == "2":
            parsed_outputs.append(1)
        else:
            parsed_outputs.append(-1)
    
    correct = [parsed == target for parsed, target in zip(parsed_outputs, targets)]
    return correct, {
        'targets': targets,
        'prompts': prompts,
        'judge_outputs': judge_responses,
        'parsed_outputs': parsed_outputs,
        'correct': correct
    }


def calculate_rewards(judge_correct: List[bool], generator_fooled_judge: List[bool]) -> Tuple[List[float], List[float]]:
    """Calculate rewards for judge and generator based on performance."""
    judge_rewards = [1.0 if correct else -0.5 for correct in judge_correct]
    generator_rewards = [1.0 if fooled else -1.0 for fooled in generator_fooled_judge]
    return judge_rewards, generator_rewards