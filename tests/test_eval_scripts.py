import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from llm_gan.eval.eval_discriminator import eval_discriminator
from llm_gan.eval.eval_generator import eval_generator


class EvalScriptsTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
        self.temp_path = handle.name
        handle.close()
        df = pd.DataFrame(
            {
                "title": ["t1", "t2"],
                "genre": ["g1", "g2"],
                "human_story": ["human story one", "human story two"],
                "ai_story": ["ai story one", "ai story two"],
            }
        )
        df.to_csv(self.temp_path, index=False)
        self.addCleanup(self._cleanup_tempfile)

    def _cleanup_tempfile(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_eval_discriminator_with_api_judge(self):
        responses = ["<answer>1</answer>", "<answer>2</answer>"]
        with patch(
            "llm_gan.utils.inference_config.batch_inference_api",
            return_value=list(responses),
        ):
            summary = eval_discriminator(
                "dummy-judge",
                dataset_path=self.temp_path,
                return_details=True,
            )
        self.assertEqual(summary["num_samples"], 2)
        self.assertIn("accuracy", summary)
        self.assertIn("details", summary)

    def test_eval_generator_uses_existing_ai_story_when_generator_missing(self):
        responses = ["<answer>2</answer>", "<answer>2</answer>"]
        with patch(
            "llm_gan.utils.inference_config.batch_inference_api",
            return_value=list(responses),
        ):
            summary = eval_generator(
                generator=None,
                judge="dummy-judge",
                dataset_path=self.temp_path,
                return_details=False,
            )
        self.assertEqual(summary["num_samples"], 2)
        self.assertIn("judge_accuracy", summary)
        self.assertIn("generator_fooling_rate", summary)


if __name__ == "__main__":
    unittest.main()
