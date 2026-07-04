"""记忆层。

记忆分类（借认知科学框架）:
  - 工作记忆   = 当前 context window（由 orchestrator 维护，不落库）
  - 情景记忆   = 发生过的对话/事件 -> 向量库
  - 语义记忆   = 关于用户的事实（职业/城市/偏好/人际）-> SQLite 结构化
  - 程序性记忆 = 用户习惯的做事方式 -> SQLite（type=procedural）

工程环节: 写入(extractor) / 存储(store) / 检索(retriever) /
          冲突更新 / 遗忘(衰减) / 巩固(consolidation)。
"""

from .models import MemoryItem, MemoryType
from .store import MemoryStore
from .extractor import MemoryExtractor
from .retriever import MemoryRetriever

__all__ = ["MemoryItem", "MemoryType", "MemoryStore", "MemoryExtractor", "MemoryRetriever"]
