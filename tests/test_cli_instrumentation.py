"""Exercise the real CLI with synthetic memory and a mocked Gemini client."""

import builtins
import io
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import errors, types


ROOT = Path(__file__).resolve().parents[1]


def run_cli(inputs, client):
    records = []
    saved_memory = []
    real_open = builtins.open

    class MemoryFile(io.StringIO):
        def close(self):
            if not self.closed:
                saved_memory.append(self.getvalue())
            super().close()

    def open_synthetic_memory(file, mode="r", *args, **kwargs):
        if file == "memory.txt":
            if mode == "w":
                return MemoryFile()
            return io.StringIO("Synthetic saved memory" if mode == "r" else "")
        return real_open(file, mode, *args, **kwargs)

    with (
        patch("builtins.input", side_effect=inputs),
        patch("builtins.open", side_effect=open_synthetic_memory),
        patch("dotenv.load_dotenv") as dotenv_mock,
        patch("os.getenv", return_value="offline-placeholder"),
        patch("google.genai.Client", return_value=client),
        patch("instrumentation.JsonlRecorder", return_value=records.append),
        patch("time.sleep"),
    ):
        state = runpy.run_path(str(ROOT / "main.py"))
    dotenv_mock.assert_called_once()
    return state, records, saved_memory


def test_cli_commands_history_and_memory_summary(capsys):
    client = Mock()
    usage = types.GenerateContentResponseUsageMetadata(
            cached_content_token_count=0, tool_use_prompt_token_count=0,
        prompt_token_count=10, candidates_token_count=2, total_token_count=12
    )
    client.models.generate_content.return_value = SimpleNamespace(
        text="Synthetic answer or summary", usage_metadata=usage
    )
    state, records, saved = run_cli(
        ["", "memory", "clear"] + [f"Question {n}" for n in range(21)] + ["exit"],
        client,
    )
    assert len(records) == 22  # 21 answers plus one memory summary.
    assert [r["request_kind"] for r in records].count("memory_summary") == 1
    assert len(state["conversation_history"]) == 40
    assert state["long_term_memory"] == "Synthetic answer or summary"
    assert saved == ["Synthetic answer or summary"]
    output = capsys.readouterr().out
    for expected in [
        "Please enter a question", "Synthetic saved memory",
        "Recent conversation history cleared", "Long-term memory saved",
        "Goodbye!",
    ]:
        assert expected in output


def test_cli_retry_still_removes_failed_message(capsys):
    client = Mock()
    client.models.generate_content.side_effect = [
        errors.ServerError(503, {"error": {"message": "Synthetic busy"}}),
        SimpleNamespace(text="Recovered", usage_metadata=None),
    ]
    state, records, saved = run_cli(["Question", "exit"], client)
    assert [r["status"] for r in records] == ["error", "success"]
    assert [message.role for message in state["conversation_history"]] == ["user", "model"]
    assert not saved
    assert "Retrying in 2 seconds" in capsys.readouterr().out
