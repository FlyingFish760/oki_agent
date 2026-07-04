"""oki — 本地个人 AI 助手。

分层架构:
    inference/  推理引擎层  (Ollama 客户端)
    persona/    人格系统    (persona card + 可调旋钮)
    memory/     记忆层      (SQLite 事实 + 向量情景记忆)
    agent/      编排/Agent  (对话主循环 + 规划 + 工具调用)
    agent/tools 工具层      (文件、日历等本地能力 + 安全 gate)
    interface/  交互层      (CLI，后续可扩展 GUI/语音)
"""

__version__ = "0.1.0"
