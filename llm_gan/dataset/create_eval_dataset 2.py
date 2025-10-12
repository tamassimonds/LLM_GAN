
from datasets import load_dataset
import pandas as pd

from llm_gan.prompts import llm_generator_prompt
from llm_gan.utils.infernece_api import batch_inference_api

def genrate_ai_stories_db(model, human_stories):

    # prompt = llm_generator_prompt.format()

    prompts = [llm_generator_prompt(title, genre) for title, genre in zip(list(human_stories["title"]), list(human_stories["genre"]))]

    respones = batch_inference_api(model, prompts)
    respones_series = pd.Series(respones)

    ai_df = respones_series.reset_index().merge(human_stories["id"].reset_index()).drop(columns=["index"])
    

    return ai_df


if __name__ == "__main__":
    
    human_stories_db = load_dataset("FareedKhan/1k_stories_100_genre")
    human_stories_df = pd.DataFrame(human_stories_db["train"])
    human_stories_df["human_story"] = human_stories_df["story"].str.split("\n").str[0]


    ai_df = genrate_ai_stories_db("openai:gpt-4.1-mini", human_stories_df).rename(columns={0:"ai_story"})
    ai_df.to_csv("data/ai_eval_stories.csv")

    comparison_db = ai_df.copy()
    comparison_db = ai_df.merge(human_stories_df)

    comparison_db.to_csv("data/gpt_4.1-mini_stories.csv")






    

