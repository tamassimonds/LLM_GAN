import unittest
from unittest.mock import patch

from llm_gan.utils.inference_config import InferenceConfig


class InferenceConfigTests(unittest.TestCase):
    def test_coerce_from_string_defaults_to_api(self):
        config = InferenceConfig.coerce("openai:gpt-4o-mini")
        self.assertEqual(config.backend, "api")
        self.assertEqual(config.model, "openai:gpt-4o-mini")

    def test_coerce_from_object_defaults_to_local(self):
        dummy_model = object()
        config = InferenceConfig.coerce(dummy_model, default_provider="huggingface")
        self.assertEqual(config.backend, "local")
        self.assertIs(config.model, dummy_model)
        self.assertEqual(config.provider, "huggingface")

    def test_run_api_backend_uses_batch_inference(self):
        config = InferenceConfig.coerce("dummy")
        with patch("llm_gan.utils.inference_config.batch_inference_api", return_value=["ok"]) as mock_batch:
            outputs = config.run(["prompt"])
        mock_batch.assert_called_once()
        self.assertEqual(outputs, ["ok"])

    def test_run_local_backend_uses_batch_local(self):
        config = InferenceConfig.coerce(
            {
                "backend": "local",
                "model": object(),
                "tokenizer": object(),
                "provider": "huggingface",
            }
        )
        with patch("llm_gan.utils.inference_config.batch_local_inference", return_value=["local"]) as mock_local:
            outputs = config.run(["prompt"])
        mock_local.assert_called_once()
        self.assertEqual(outputs, ["local"])


if __name__ == "__main__":
    unittest.main()
