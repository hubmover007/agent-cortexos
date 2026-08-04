"""冲突消解 —— 同 subject+predicate 事实的时间窗口截断。

严格按照 code-design-p1.md §3.5 伪代码实现。
"""

from __future__ import annotations

import time
from typing import List, Optional

from cortexos.config import Config
from cortexos.models import Fact


async def resolve_fact(
    new_fact: Fact,
    existing_facts: List[Fact],
    config: Config,
    now: Optional[float] = None,
) -> Fact:
    """对同 scope + subject+predicate 的活跃事实做冲突消解。

    规则（code-design-p1.md §3.5）：
    1. object 一致 → 合并：置信度取高，valid_until 取远
    2. object 矛盾 + new.valid_from >= old.valid_from → 时间窗口截断
       old.valid_until = new.valid_from, old.status = superseded/active
    3. 新旧无法判断 → 两条都保留，new.status = conflict

    Args:
        new_fact: 新事实。
        existing_facts: 已有活跃事实列表。
        config: 配置。
        now: 当前时间戳。

    Returns:
        处理后的 new_fact（status 可能被修改）。
    """
    if now is None:
        now = time.time()

    if not config.resolve.enabled:
        return new_fact

    for old in existing_facts:
        if old.subject != new_fact.subject or old.predicate != new_fact.predicate:
            continue
        if old.id == new_fact.id:
            continue

        if old.object == new_fact.object:
            # 一致 → 合并：置信度取高；有效期取“更远”（None=永不过期优先）
            old.confidence = max(old.confidence, new_fact.confidence)
            if new_fact.valid_until is None:
                old.valid_until = None
            elif old.valid_until is not None:
                old.valid_until = max(old.valid_until, new_fact.valid_until)
            new_fact.status = "superseded"
        else:
            # 矛盾 → 时间窗口截断
            if new_fact.valid_from >= old.valid_from:
                old.valid_until = new_fact.valid_from
                if old.valid_until is not None and old.valid_until > now:
                    old.status = "active"
                else:
                    old.status = "superseded"
                new_fact.status = "active"
            else:
                # 无法判断新旧 → 都保留
                new_fact.status = "conflict"

    return new_fact
