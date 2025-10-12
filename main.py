from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model_name = "gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
generator_model = GPT2LMHeadModel.from_pretrained(model_name)
discriminator_model = GPT2LMHeadModel.from_pretrained(model_name)



"""
How dow e train each?
it's very easy for the dirimator to randomly guess the right answer so very noisy


We generate a bunch then we train the genrator and discrimator at the same time 



"""







