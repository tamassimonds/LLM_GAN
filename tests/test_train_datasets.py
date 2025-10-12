import torch

from llm_gan.train.dataset import GeneratorDataset, JudgeDataset


class DummyTokenizer:
    def __init__(self):
        self.word_to_id = {"<pad>": 0, "<eos>": 1}
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def _tokenize(self, text):
        tokens = text.replace("\n", " ").split()
        ids = []
        for token in tokens:
            if token not in self.word_to_id:
                self.word_to_id[token] = len(self.word_to_id)
            ids.append(self.word_to_id[token])
        return ids

    def __call__(
        self,
        text,
        *,
        max_length=None,
        truncation=False,
        return_tensors=None,
        add_special_tokens=True,
    ):
        ids = self._tokenize(text)
        if add_special_tokens:
            ids = ids + [self.word_to_id[self.eos_token]]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        raise ValueError("Only return_tensors='pt' supported in DummyTokenizer")

    def __len__(self):
        return len(self.word_to_id)


def test_generator_dataset_masks_prompt():
    import pandas as pd

    df = pd.DataFrame(
        {
            "title": ["Title"],
            "genre": ["Genre"],
            "human_story": ["This is a story"],
        }
    )
    tokenizer = DummyTokenizer()
    dataset = GeneratorDataset(df, tokenizer, max_length=64)
    item = dataset[0]
    assert item["input_ids"].shape == item["attention_mask"].shape
    assert (item["labels"] == -100).any(), "prompt tokens should be masked"


def test_judge_dataset_returns_labels():
    import pandas as pd

    df = pd.DataFrame(
        {
            "title": ["Title"],
            "genre": ["Genre"],
            "human_story": ["Human story"],
            "ai_story": ["AI story"],
        }
    )
    tokenizer = DummyTokenizer()
    dataset = JudgeDataset(df, tokenizer, max_length=64)
    item = dataset[0]
    assert "labels" in item
    assert item["labels"].dtype == torch.long
    assert item["input_ids"].ndim == 1
