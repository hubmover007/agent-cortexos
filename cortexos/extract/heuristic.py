"""启发式提取 —— 无 LLM 降级方案。

使用关键词匹配和正则表达式提取实体和事实。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# 常见技术实体模式
_TECH_PATTERNS = [
    r"\b(Kubernetes|K8s|k8s)\b",
    r"\b(Docker|docker)\b",
    r"\b(Pod|pod|POD)\b",
    r"\b(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Kafka|RabbitMQ)\b",
    r"\b(Nginx|nginx|NGINX|HAProxy|haproxy)\b",
    r"\b(AWS|GCP|Azure|阿里云|腾讯云)\b",
    r"\b(ECS|EC2|S3|RDS|Lambda|lambda)\b",
    r"\b(Terraform|terraform|Ansible|ansible|Pulumi)\b",
    r"\b(GitHub|GitLab|Jenkins|CircleCI)\b",
    r"\b(OpenClaw|Agent|Bot|bot|Gateway|gateway)\b",
    r"\b(API|api|REST|GraphQL|gRPC|WebSocket)\b",
    r"\b(OOM|CPU|内存|磁盘|网络|带宽|延迟|latency)\b",
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP 地址
    r"\b(i-[a-f0-9]{8,17})\b",  # AWS 实例 ID
    r"\b(iZ[a-z0-9]+)\b",       # 阿里云实例 ID
]


def heuristic_extract(content: str) -> Dict[str, Any]:
    """从文本中启发式提取实体和关键词。

    不依赖 LLM，纯正则 + 关键词匹配。

    Args:
        content: 输入文本。

    Returns:
        {
            "entities": ["实体1", "实体2", ...],
            "keywords": ["关键词1", ...],
            "triples": [],  # 启发式不支持三元组提取
        }
    """
    entities: List[str] = []
    seen = set()

    for pattern in _TECH_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            entity = match.group(0).strip()
            if entity and entity not in seen:
                seen.add(entity)
                entities.append(entity)

    return {
        "entities": entities,
        "keywords": entities[:],  # 关键词同实体
        "triples": [],
    }
