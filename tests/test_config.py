import os
import tempfile
import unittest
from pathlib import Path

from app.config import load_env_file


class ConfigTest(unittest.TestCase):
    def test_load_env_file_reads_key_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# ignored",
                        "OPENAI_API_KEY=test-key",
                        'OPENAI_MODEL="gpt-test"',
                        "INVALID_LINE",
                    ]
                ),
                encoding="utf-8",
            )

            old_key = os.environ.pop("OPENAI_API_KEY", None)
            old_model = os.environ.pop("OPENAI_MODEL", None)
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["OPENAI_MODEL"], "gpt-test")
            finally:
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("OPENAI_MODEL", None)
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key
                if old_model is not None:
                    os.environ["OPENAI_MODEL"] = old_model

    def test_load_env_file_does_not_override_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=file-key", encoding="utf-8")

            old_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "shell-key"
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-key")
            finally:
                if old_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
