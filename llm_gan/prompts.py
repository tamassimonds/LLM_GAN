



def llm_generator_discriminator_prompt(title, genre, story1, story2): 
    return f""" 
    <USER>
    You will be given two sample outputs. Your goal is to select which one if formulated by a human and which one by ai
    title: {title}
    genre: {genre}

    STORY 1
    {story1}

    STORY 2
    {story2}

    Reason first then output  1 or 2 in answer tags e.g corresponding to story 1 or 2
    Example output: <answer>1</answer>
    </USER>
    <ASSISTANT>
    """

def llm_generator_prompt(title, genre): 
    return f"""
    <USER>
    Your goal is to genreate a story that is as human like as possible. You may plan before writting

    Then write in the tags below output your final story

    The sample title is {title}

    The genre of the story is {genre}

    <story>Put your final story inside story tags</story>
    </USER>
    <ASSISTANT>
    """
