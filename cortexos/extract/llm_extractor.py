"""LLM 提取器 —— OpenAI 兼容 Chat API 提取实体/事实/关系。

使用 LLM 从文本中提取结构化信息：
- 实体（entities）
- 结构化事实三元组（facts）
- 实体间关系（edges）
- 时间线索（valid_until）

提取失败时降级为启发式（无 LLM 模式）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from cortexos.config import LLMConfig
from cortexos.embedding.openai_compat import OpenAICompatEmbedder
from cortexos.extract.heuristic import heuristic_extract

_EXTRACT_PROMPT = """你是一个信息提取助手。从以下文本中提取结构化信息，返回 JSON。

## 提取要求
1. entities: 从文本中提取的关键实体列表（技术名词、服务名、错误类型、配置项等）
2. facts: 三元组列表，每个包含 subject（主语）、predicate（谓词）、object（宾语）
3. edges: 实体间关系列表，每个包含 source（源实体）、target（目标实体）、relation（关系描述）
4. valid_until: 如果文本暗示临时性（如"临时"、"暂定"、"下月"），推断过期时间

## 输出格式（JSON）
{{
  "entities": ["实体1", "实体2"],
  "facts": [
    {{"subject": "主语", "predicate": "谓词", "object": "宾语"}}
  ],
  "edges": [
    {{"source": "源实体", "target": "目标实体", "relation": "关系描述"}}
  ],
  "valid_until": null
}}

## 文本
{content}

请只返回 JSON，不要其他内容。"""


class LLMExtractor:
    """使用 LLM 从文本中提取结构化信息。

    封装 OpenAI 兼容 Chat API，可配置关闭（降级为启发式）。
    """

    def __init__(
        self,
        embedder: Optional[OpenAICompatEmbedder] = None,
        config: Optional[LLMConfig] = None,
    ):
        """初始化 LLM 提取器。

        Args:
            embedder: OpenAI 兼容 Embedder（用于 Chat API）。
            config: LLM 配置。
        """
        self._embedder = embedder
        self._config = config or LLMConfig()

    @property
    def is_available(self) -> bool:
        """LLM 是否可用。"""
        return self._embedder is not None and self._embedder.is_available

    async def extract(self, content: str) -> Dict[str, Any]:
        """从文本中提取结构化信息。

        LLM 可用时调用 API，不可用时降级为启发式。

        Args:
            content: 输入文本。

        Returns:
            {
                "entities": [...],
                "facts": [...],
                "edges": [...],
                "valid_until": null | float,
            }
        """
        # 配置关闭或 LLM 不可用 → 降级
        if not self.is_available or not self._config.extract_entities:
            return {
                **heuristic_extract(content),
                "facts": [],
                "edges": [],
                "valid_until": None,
            }

        try:
            prompt = _EXTRACT_PROMPT.format(content=content[:3000])
            result = await self._embedder.chat_json(  # type: ignore[union-attr]
                messages=[{"role": "user", "content": prompt}],
            )

            if "error" in result:
                # JSON 解析失败 → 降级
                return {
                    **heuristic_extract(content),
                    "facts": [],
                    "edges": [],
                    "valid_until": None,
                }

            entities = result.get("entities", [])
            facts = result.get("facts", []) if self._config and self._config.extract_facts else []
            edges = result.get("edges", []) if self._config and self._config.extract_edges else []

            valid_until = result.get("valid_until")
            if isinstance(valid_until, str) and valid_until:
                # 尝试解析 LLM 返回的日期字符串
                try:
                    valid_until = _parse_valid_until(valid_until)
                except Exception:
                    valid_until = None
            elif not isinstance(valid_until, (int, float)):
                valid_until = None

            return {
                "entities": entities if isinstance(entities, list) else [],
                "facts": facts if isinstance(facts, list) else [],
                "edges": edges if isinstance(edges, list) else [],
                "valid_until": valid_until,
            }
        except Exception:
            # 任何异常 → 降级
            return {
                **heuristic_extract(content),
                "facts": [],
                "edges": [],
                "valid_until": None,
            }


def _parse_valid_until(date_str: str) -> float:
    """尝试解析 LLM 返回的日期字符串为 epoch 秒。

    当前简化实现：将 "下月" "明天" 等转为相对时间。
    """
    # 尝试 ISO 格式解析
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        pass

    # 相对时间关键词
    now = time.time()
    if "天" in date_str or "day" in date_str.lower():
        # 提取数字
        import re
        num = re.findall(r"\d+", date_str)
        days = int(num[0]) if num else 1
        return now + days * 86400
    if "小时" in date_str or "hour" in date_str.lower():
        import re
        num = re.findall(r"\d+", date_str)
        hours = int(num[0]) if num else 1
        return now + hours * 3600

    # 默认 7 天
    return now + 7 * 86400
