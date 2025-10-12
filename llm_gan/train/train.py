
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

from llm_gan.prompts import llm_generator_prompt, llm_generator_discriminator
from llm_gan.utils import batch_local_inference
from .dataset import Dataset


"""
Guess this doesn't super make sense seeing that we aern't doing GRPO?
"""
# def genereate_rollout(model, prompt: list[str], n_rollouts) -> list[str]:

#     pas


def asses_judge(titles, genres, stories_human, stories_ai) -> list[bool]:

    targets = [random.randint(0,1) for _ in range(len(titles))]

    prompts = []
    for title, genre, human_story, ai_story, target in zip(titles, genres, stories_human, stories_ai, targets):
        story_order = (human_story, ai_story) if target == 0 else (ai_story, human_story)
        prompt = llm_generator_discriminator_prompt(title, genre, *story_order)
        prompts.append(prompt)
    
    judge_ouputs = batch_local_inference(prompts)
    parsed_outputs = (judge_ouputs)

    correct = [parsed == target for parsed,target in zip(parsed_outputs, targets)]
    return correct
    
    



bs = 64
epochs = 100

dataloader = torch.utils.data.DataLoader(Dataset, batch_size=64)

model_name = "gpt2"
generator_model = generator_model = GPT2LMHeadModel.from_pretrained(model_name)
judge_model  = generator_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)





for _ in range(epochs):
    for inputs, targets in dataloader:
        
        generator_prompts = [llm_generator_discriminator(input["title"], input["genre"]) for input in inputs]

        orginal_stories = inputs["orginal_stories"]
        
        generator_stories = batch_local_inference(generator_prompts)

        #need to handle randomizing stories somewhere

        judge_results = asses_judge(input["title"], input["genre"], orginal_stories, generator_stories)

        generator_results = 

        


        

        
        










