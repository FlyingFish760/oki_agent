"""Standard-library smoke test for environments without pytest."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oki.agent import OkiAgent
from oki.memory import MemoryStore
from oki.models import DemoClient
from oki.tools import ToolRegistry


def main() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        store = MemoryStore(root / "memory.sqlite3")
        agent = OkiAgent(DemoClient("persona"), store, ToolRegistry(root))
        first = "".join(agent.stream_turn("我喜欢简短直接的解释，以后记住"))
        assert "稳稳接住" in first
        assert store.list(), "expected at least one memory"
        result = ToolRegistry(root).maybe_run("帮我删除所有临时文件")
        assert result is not None and result.requires_confirmation
    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


