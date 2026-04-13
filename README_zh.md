# Agent-CortexOS

> 通用 AI Agent 认知操作系统 — 结构化记忆、跨会话状态恢复、主动任务推进。

## 解决什么问题

当前所有 AI Agent 面临三个根本性缺陷：

- **记忆是流水账** — 所有信息按时间线性堆叠，无法按"知识领域"组织
- **每次会话失忆** — 新会话 = 从零开始，知识无法跨会话沉淀
- **任务不会推进** — Agent 只响应当前指令，不会主动跟进未完成事项

## 快速开始

### 安装

```bash
pip install agent-cortexos
```

### 5 行代码体验

```python
import cortexos

cx = cortexos.init()
cx.store("K8s Pod 重启问题：检查 OOMKilled，调整 memory limit")
cx.store("Docker 镜像构建优化：多阶段构建可减少 60% 镜像体积")
results = cx.recall("容器内存问题")
print(results[0].content)
```

### CLI 体验

```bash
# 存储记忆
cortexos store "会议决定：Q2 迁移到 K8s" --type decision

# 检索记忆
cortexos recall "Q2 计划"

# 查看 Zone 列表
cortexos zones list

# 系统统计
cortexos stats

# 任务管理
cortexos task create "完成 API 文档" --due 2025-04-30
cortexos task list
```

## 核心架构

### Zone（知识领域）

Zone 是记忆的"引力场"。每条记忆通过三层路由自动归入最匹配的 Zone：

1. **实体反查** — 记忆中的实体已绑定到某 Zone → 直接路由
2. **TF-IDF scope 匹配** — 内容与 Zone scope 做相似度计算
3. **兜底** — 未命中的进入 `_inbox`，积累到阈值后自动涌现新 Zone

### 检索公式

```
score = 0.4 × 文本相似度 + 0.3 × 时效性 + 0.2 × Zone引力 + 0.1 × 访问频率
```

### 三层处理架构

- **实时层** (< 50ms): store / recall / session_context
- **近线层** (秒级): Zone 涌现、去重、scope 更新
- **离线层** (分钟级): gravity 衰减、归档、TF-IDF 重建

**铁律：主 Agent 永不阻塞。**

## 路线图

- **v0.1**（当前）— JSONL 存储、TF-IDF 检索、Zone 路由、任务系统、CLI
- **v0.2** — 向量嵌入支持
- **v0.3** — 多 Agent 协作
- **v0.5** — SQLite/PostgreSQL 后端
- **v1.0** — 生产就绪：REST API、Docker、插件系统

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
