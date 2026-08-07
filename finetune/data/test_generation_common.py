from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from .generation_common import (
    TeacherPrompt,
    call_openai_compatible,
    configure_teacher_args,
)


class OpenAICompatibleTest(unittest.TestCase):
    def test_calls_chat_completions_and_parses_json(self) -> None:
        prompt = TeacherPrompt(
            system="Generate data.",
            user="Return one JSON response.",
        )
        args = argparse.Namespace(
            model="local-model",
            base_url="http://127.0.0.1:8000/v1/",
            api_key="local-key",
            json_response_format=True,
            temperature=0.7,
            top_p=0.9,
        )
        endpoint_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"response": "Hello"}',
                    }
                }
            ]
        }

        with patch(
            "finetune.data.generation_common._post_json",
            return_value=endpoint_response,
        ) as post_json:
            result = call_openai_compatible(prompt, args)

        self.assertEqual(result, {"response": "Hello"})
        post_json.assert_called_once_with(
            "http://127.0.0.1:8000/v1/chat/completions",
            {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "Generate data."},
                    {
                        "role": "user",
                        "content": "Return one JSON response.",
                    },
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": "Bearer local-key"},
        )

    def test_requires_base_url(self) -> None:
        args = argparse.Namespace(
            teacher="openai-compatible",
            model="local-model",
            base_url="",
        )

        with self.assertRaisesRegex(ValueError, "--base-url is required"):
            configure_teacher_args(args, require_model=True)


if __name__ == "__main__":
    unittest.main()
