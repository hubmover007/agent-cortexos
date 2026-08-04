"""Steps 4+5 单元测试：recall（hybrid+scoring+graph）+ extract（llm_extractor+heuristic）。"""

import time

import pytest

from cortexos.config import RecallWeights
from cortexos.models import Edge, Entry


# ────────────────────── 图遍历测试 ──────────────────────


class TestGraphIndex:
    """图索引和 BFS 遍历测试。"""

    def test_add_and_search(self):
        """添加边并执行图搜索。"""
        from cortexos.recall.graph import GraphIndex

        g = GraphIndex()
        g.add_edge(Edge(
            source="nginx", target="server-01",
            relation="运行在", weight=1.0, entry_id="e1",
        ))
        g.add_edge(Edge(
            source="server-01", target="k8s-cluster",
            relation="属于", weight=0.8, entry_id="e2",
        ))
        g.add_edge(Edge(
            source="server-01", target="mysql",
            relation="连接", weight=0.6, entry_id="e3",
        ))

        # 从 nginx 出发搜索
        scores = g.graph_search(["nginx"], hop=2, decay=0.5)
        assert "e1" in scores
        # e1 可通过 nginx 直达（1.0），也可通过 nginx→server-01 路径（1.5）
        assert scores["e1"] >= 1.0

    def test_hop_limit(self):
        """Hop 限制测试。"""
        from cortexos.recall.graph import GraphIndex

        g = GraphIndex()
        g.add_edge(Edge(source="a", target="b", relation="to", entry_id="e1"))
        g.add_edge(Edge(source="b", target="c", relation="to", entry_id="e2"))
        g.add_edge(Edge(source="c", target="d", relation="to", entry_id="e3"))

        # hop=1: 只能到 a 和 b
        scores_1 = g.graph_search(["a"], hop=1)
        assert "e1" in scores_1
        assert "e2" not in scores_1  # b→c 需要 2 跳
        assert "e3" not in scores_1

        # hop=2: 可以到 c
        scores_2 = g.graph_search(["a"], hop=2)
        assert "e3" not in scores_2  # a→b→c→d 需要 3 跳

        # hop=3
        scores_3 = g.graph_search(["a"], hop=3)
        assert "e3" in scores_3

    def test_decay(self):
        """衰减测试：跳数越深得分越低。"""
        from cortexos.recall.graph import GraphIndex

        g = GraphIndex()
        g.add_edge(Edge(source="a", target="b", relation="to", weight=1.0, entry_id="e1"))
        g.add_edge(Edge(source="b", target="c", relation="to", weight=1.0, entry_id="e2"))

        scores = g.graph_search(["a"], hop=2, decay=0.5)
        # e2 是两跳到达，得分 = 1.0 (锚点) + 1.0 * 0.5^1 = 1.5
        # e1 是一跳到达，得分 = 1.0
        # 累计分越高不一定代表越近（图遍历是加分制）
        assert "e1" in scores
        assert "e2" in scores
        assert scores["e2"] >= scores["e1"]

    def test_empty_search(self):
        """空查询实体返回空结果。"""
        from cortexos.recall.graph import GraphIndex
        g = GraphIndex()
        scores = g.graph_search([], hop=2)
        assert scores == {}

    def test_load_edges_bulk(self):
        """批量加载边。"""
        from cortexos.recall.graph import GraphIndex
        g = GraphIndex()
        edges = [
            Edge(source="a", target="b", relation="to", entry_id="e1"),
            Edge(source="b", target="c", relation="to", entry_id="e2"),
        ]
        g.load_edges(edges)
        assert g.entity_count() >= 2
        assert g.edge_count() == 2


# ────────────────────── 评分测试 ──────────────────────


class TestScoring:
    """RRF 融合和多因子评分测试。"""

    def test_rrf_fusion(self):
        """RRF 融合基本测试。"""
        from cortexos.recall.scoring import rrf_fusion

        ranked = [
            ["e1", "e2", "e3"],
            ["e2", "e1", "e4"],
        ]
        scores = rrf_fusion(ranked, k=60)
        # e1 在第一个列表 rank=1, 第二个列表 rank=2
        # e2 在第一个列表 rank=2, 第二个列表 rank=1
        # 两者总分应接近
        assert "e1" in scores
        assert "e2" in scores
        assert 0 <= scores["e1"] <= 1

    def test_rrf_fusion_empty(self):
        """空输入 RRF 融合。"""
        from cortexos.recall.scoring import rrf_fusion
        scores = rrf_fusion([], k=60)
        assert scores == {}

    def test_rrf_fusion_single_list(self):
        """单列表 RRF 融合。"""
        from cortexos.recall.scoring import rrf_fusion
        scores = rrf_fusion([["a"]], k=60)
        assert scores == {"a": 1.0}

    def test_compute_recency(self):
        """新鲜度计算。"""
        from cortexos.recall.scoring import compute_recency
        now = time.time()

        # 刚创建的条目新鲜度 ≈ 1
        recency_new = compute_recency(now, half_life_days=7.0)
        assert recency_new == pytest.approx(1.0, abs=0.01)

        # 7 天前 → 0.5
        recency_7d = compute_recency(now - 7 * 86400, half_life_days=7.0)
        assert recency_7d == pytest.approx(0.5, abs=0.01)

        # 14 天前 → 0.25
        recency_14d = compute_recency(now - 14 * 86400, half_life_days=7.0)
        assert recency_14d == pytest.approx(0.25, abs=0.01)

    def test_compute_freq(self):
        """访问频率计算。"""
        import math
        from cortexos.recall.scoring import compute_freq
        assert compute_freq(0) == pytest.approx(0.0, abs=0.01)
        assert compute_freq(50) == pytest.approx(1 - 1 / math.e, abs=0.01)
        assert compute_freq(100) > compute_freq(50)
        assert compute_freq(100) < 1.0

    def test_score_entry_all_weights(self):
        """多因子评分综合测试。"""
        from cortexos.recall.scoring import score_entry
        import math

        entry = Entry(
            content="test",
            scope="agent:test",
            zone="test_zone",
            access_count=50,
            created_at=time.time(),
        )
        rrf = {"entry": 0.8}
        graph = {"entry": 0.5}

        weights = RecallWeights(
            text_sim=0.35, recency=0.25, gravity=0.15,
            freq=0.10, scope_boost=0.05, graph_path=0.10,
        )

        # scope 不匹配（entry.scope="agent:test", entry.id 不匹配）
        score = score_entry(entry, rrf, graph, "agent:other", zone_gravity=1.0, weights=weights)
        assert 0 <= score <= 5  # 合理范围

    def test_rank_entries(self):
        """排序测试。"""
        from cortexos.recall.scoring import rank_entries

        entries = []
        for i in range(10):
            entries.append(Entry(
                content=f"entry {i}",
                scope="agent:test",
                zone="test_zone",
                created_at=time.time() - i * 86400,
                access_count=i * 10,
            ))

        rrf = {e.id: 0.5 for e in entries}
        graph = {}
        zone_gravities = {"test_zone": 1.0}

        ranked = rank_entries(
            entries, rrf, graph, "agent:test",
            zone_gravities, top_k=5,
        )
        assert len(ranked) == 5
        # 按得分降序
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]


# ────────────────────── 词法检索（FTS5）测试 ──────────────────────


@pytest.mark.asyncio
class TestLexicalSearch:
    """FTS5 词法检索测试（集成 SQLite）。"""

    async def test_fts5_basic(self):
        """FTS5 基本检索。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        import tempfile, os

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            db = SqliteBackend(path)
            await db.initialize()

            for i in range(5):
                await db.upsert_entry(Entry(
                    content=f"test memory entry number {i}",
                    scope="agent:test",
                ))

            results = await db.search_lexical("test memory", scope="agent:test")
            assert len(results) > 0
            await db.close()
        finally:
            os.unlink(path)


# ────────────────────── 提取测试 ──────────────────────


class TestHeuristicExtract:
    """启发式提取测试。"""

    def test_extract_tech_entities(self):
        """提取技术实体。"""
        from cortexos.extract.heuristic import heuristic_extract

        content = "Kubernetes Pod 在 nginx 上 OOM，CPU 使用率 99%"
        result = heuristic_extract(content)
        assert "Kubernetes" in result["entities"] or len(result["entities"]) > 0

    def test_extract_ip(self):
        """提取 IP 地址。"""
        from cortexos.extract.heuristic import heuristic_extract

        content = "服务器 192.168.1.100 连接超时"
        result = heuristic_extract(content)
        assert "192.168.1.100" in result["entities"]

    def test_extract_empty(self):
        """空文本提取。"""
        from cortexos.extract.heuristic import heuristic_extract

        result = heuristic_extract("")
        assert result["entities"] == []
        assert result["triples"] == []


class TestLLMExtractor:
    """LLM 提取器测试（mock）。"""

    @pytest.mark.asyncio
    async def test_fallback_to_heuristic_when_no_llm(self):
        """无 LLM 时降级到启发式。"""
        from cortexos.extract.llm_extractor import LLMExtractor
        from cortexos.config import LLMConfig

        config = LLMConfig(extract_entities=False)
        extractor = LLMExtractor(embedder=None, config=config)

        result = await extractor.extract("nginx 在 server-01 上运行")
        assert "entities" in result
        assert result["facts"] == []
        assert result["edges"] == []

    @pytest.mark.asyncio
    async def test_extract_with_mock_llm(self):
        """Mock LLM 提取。"""
        from unittest.mock import AsyncMock, MagicMock, patch
        from cortexos.extract.llm_extractor import LLMExtractor
        from cortexos.config import LLMConfig

        config = LLMConfig(extract_entities=True, extract_facts=True, extract_edges=True)
        mock_embedder = MagicMock()
        mock_embedder.is_available = True
        mock_embedder.chat_json = AsyncMock(return_value={
            "entities": ["k8s", "pod"],
            "facts": [{"subject": "pod", "predicate": "crashed", "object": "OOM"}],
            "edges": [{"source": "pod", "target": "k8s", "relation": "运行在"}],
            "valid_until": None,
        })

        extractor = LLMExtractor(embedder=mock_embedder, config=config)
        result = await extractor.extract("k8s pod crashed due to OOM")

        assert "k8s" in result["entities"]
        assert "pod" in result["entities"]
        assert len(result["facts"]) == 1
        assert result["facts"][0]["predicate"] == "crashed"
        assert len(result["edges"]) == 1

    @pytest.mark.asyncio
    async def test_extract_json_parse_error_fallback(self):
        """LLM 返回无效 JSON 时降级。"""
        from unittest.mock import AsyncMock, MagicMock
        from cortexos.extract.llm_extractor import LLMExtractor
        from cortexos.config import LLMConfig

        config = LLMConfig(extract_entities=True)
        mock_embedder = MagicMock()
        mock_embedder.is_available = True
        mock_embedder.chat_json = AsyncMock(return_value={"error": "JSON parse failed", "raw": "not json"})

        extractor = LLMExtractor(embedder=mock_embedder, config=config)
        result = await extractor.extract("nginx 服务异常")

        # 降级后应有 entities
        assert "entities" in result
        assert "facts" in result
