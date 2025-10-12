import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from llm_gan.prompts import llm_generator_prompt, llm_generator_discriminator_prompt
from llm_gan.utils import batch_local_inference
from llm_gan.utils.parse import parse_tags


class LLMGANInference:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
                 generator_checkpoint: str = None,
                 judge_checkpoint: str = None):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if generator_checkpoint:
            self.generator_model.load_state_dict(torch.load(generator_checkpoint))
            
        if judge_checkpoint:
            self.judge_model.load_state_dict(torch.load(judge_checkpoint))
            
        self.generator_model.eval()
        self.judge_model.eval()
    
    def generate_story(self, title: str, genre: str, max_length: int = 200) -> str:
        prompt = llm_generator_prompt(title, genre)
        
        response = batch_local_inference(
            [prompt],
            model=self.generator_model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_length,
            temperature=0.8,
            do_sample=True
        )[0]
        
        story = parse_tags(response, "story")
        if story is None:
            story = response.split("ASSISTANT>")[-1].strip()
        
        return story
    
    def judge_stories(self, title: str, genre: str, story1: str, story2: str) -> dict:
        prompt = llm_generator_discriminator_prompt(title, genre, story1, story2)
        
        response = batch_local_inference(
            [prompt],
            model=self.judge_model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            temperature=0.3
        )[0]
        
        answer = parse_tags(response, "answer")
        if answer == "1":
            prediction = 1
        elif answer == "2":
            prediction = 2
        else:
            prediction = None
        
        return {
            "prediction": prediction,
            "reasoning": response.split("ASSISTANT>")[-1].strip() if "ASSISTANT>" in response else response
        }
    
    def interactive_demo(self):
        print("LLM GAN Interactive Demo")
        print("Commands: 'generate' or 'judge' or 'quit'")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "quit":
                break
            elif command == "generate":
                title = input("Enter story title: ")
                genre = input("Enter genre: ")
                story = self.generate_story(title, genre)
                print(f"\nGenerated Story:\n{story}")
            elif command == "judge":
                title = input("Enter story title: ")
                genre = input("Enter genre: ")
                story1 = input("Enter first story: ")
                story2 = input("Enter second story: ")
                result = self.judge_stories(title, genre, story1, story2)
                print(f"\nJudge Result:")
                print(f"Prediction: Story {result['prediction']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Reasoning: {result['reasoning']}")
            else:
                print("Unknown command. Use 'generate', 'judge', or 'quit'")


if __name__ == "__main__":
    inference = LLMGANInference()
    inference.interactive_demo()