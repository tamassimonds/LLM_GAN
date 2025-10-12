import unittest

from llm_gan.eval.common import extract_tagged_text, parse_answer_tag, run_pairwise_judge
from llm_gan.utils.inference_config import InferenceConfig


class DummyJudge(InferenceConfig):
    def __init__(self, responses):
        super().__init__(backend="api", model="stub")
        self._responses = responses

    def run(self, prompts):  # type: ignore[override]
        self.prompts = prompts
        return list(self._responses)


class EvalCommonTests(unittest.TestCase):
    def test_extract_tagged_text(self):
        text = "<story> Hello </story>"
        self.assertEqual(extract_tagged_text(text, "story"), "Hello")
        self.assertEqual(extract_tagged_text("No tags", "story"), "No tags")

    def test_parse_answer_tag(self):
        self.assertEqual(parse_answer_tag("<answer>1</answer>"), 1)
        self.assertEqual(parse_answer_tag("Answer: 2"), 2)
        self.assertIsNone(parse_answer_tag("No numbers here"))

    def test_run_pairwise_judge(self):
        judge = DummyJudge(["<answer>1</answer>", "<answer>2</answer>"])
        result = run_pairwise_judge(
            human_stories=["human one", "human two"],
            ai_stories=["ai one", "ai two"],
            titles=["t1", "t2"],
            genres=["g1", "g2"],
            judge=judge,
            seed=0,
            return_details=True,
        )
        self.assertEqual(result.num_samples, 2)
        self.assertGreaterEqual(result.accuracy, 0)
        self.assertEqual(result.invalid_responses, 0)
        self.assertEqual(len(result.details), 2)


if __name__ == "__main__":
    unittest.main()
