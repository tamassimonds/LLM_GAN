import unittest

from llm_gan.utils.infernece_api import batch_inference_api
from llm_gan.utils.infernece_local import batch_local_inference

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for tests
    torch = None


if torch is not None:

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, texts, return_tensors="pt", padding=True):
            assert return_tensors == "pt"
            lengths = [len(t) + 1 for t in texts]  # +1 to account for eos token
            max_len = max(lengths)
            input_ids = []
            for length in lengths:
                tokens = [1] * (length - 1) + [self.eos_token_id]
                if padding:
                    tokens += [self.pad_token_id] * (max_len - length)
                input_ids.append(tokens)
            ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = (ids_tensor != self.pad_token_id).long()
            return {"input_ids": ids_tensor, "attention_mask": attention_mask}

        def batch_decode(self, sequences, skip_special_tokens=True):
            outputs = []
            for seq in sequences.tolist():
                outputs.append(f"generated-{seq[-1]}")
            return outputs


    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
            batch_size = input_ids.shape[0]
            new_tokens = torch.full(
                (batch_size, 1), 9, dtype=input_ids.dtype, device=input_ids.device
            )
            return torch.cat([input_ids, new_tokens], dim=1)

else:  # pragma: no cover - exercised only when torch missing
    DummyTokenizer = DummyModel = None


@unittest.skipUnless(torch is not None, "torch is required for local inference tests")
class BatchLocalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer()

    def test_generates_outputs_for_prompts(self):
        prompts = ["hello", "world"]
        outputs = batch_local_inference(
            prompts,
            model=self.model,
            tokenizer=self.tokenizer,
            provider="huggingface",
            batch_size=1,
            max_tokens=1,
        )
        self.assertEqual(outputs, ["generated-9", "generated-9"])

    def test_requires_model_and_tokenizer(self):
        with self.assertRaises(ValueError):
            batch_local_inference(["hi"], tokenizer=self.tokenizer)
        with self.assertRaises(ValueError):
            batch_local_inference(["hi"], model=self.model)


class InferenceAPILocalRoutingTests(unittest.TestCase):
    def test_local_provider_prefix_raises_guidance(self):
        with self.assertRaises(ValueError) as ctx:
            batch_inference_api("local:dummy-model", ["hi there"])
        self.assertIn("batch_local_inference", str(ctx.exception))

    def test_local_provider_option_raises_guidance(self):
        with self.assertRaises(ValueError) as ctx:
            batch_inference_api("dummy-model", ["hi"], provider="local_vllm")
        self.assertIn("batch_local_inference", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
