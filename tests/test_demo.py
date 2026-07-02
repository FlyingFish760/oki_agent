from pathlib import Path

from oki.agent import OkiAgent
from oki.memory import MemoryStore
from oki.models import DemoClient
from oki.tools import ToolRegistry


def test_memory_roundtrip(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    memory_id = store.add("preference", "我喜欢简短直接的解释", "seed", 4)
    assert memory_id > 0
    assert store.search("解释")
    assert store.delete(memory_id)
    assert not store.search("解释")


def test_agent_offline_turn_writes_preference(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    agent = OkiAgent(DemoClient("persona"), store, ToolRegistry(tmp_path))
    output = "".join(agent.stream_turn("我喜欢简短直接的解释，以后记住"))
    assert "稳稳接住" in output
    memories = store.list()
    assert len(memories) >= 1
    assert memories[0].type in {"preference", "procedure"}


def test_tool_safety_gate(tmp_path: Path):
    result = ToolRegistry(tmp_path).maybe_run("帮我删除所有临时文件")
    assert result is not None
    assert result.requires_confirmation
