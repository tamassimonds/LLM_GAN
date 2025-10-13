



def llm_generator_discriminator_prompt(title, genre, story1, story2): 
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

You need to determine which story is better

Title: {title}
Genre: {genre}

STORY 1:
{story1}

STORY 2:
{story2}


Give your reasoning briefly, then answer with <answer>1</answer> or <answer>2</answer> to indicate which story is better.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def llm_generator_prompt(title, genre): 
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Write a creative story in the {genre} genre with the title "{title}". 

Make the story feel human-written and engaging. Write approximately 200-300 words.

Put your story inside <story> tags like this:
<story>Your story here</story><|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
