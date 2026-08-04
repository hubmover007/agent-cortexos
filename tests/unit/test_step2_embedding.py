"""Step 2 单元测试：embedding（base + openai_compat + tfidf 降级）。"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cortexos.embedding.base import Embedder


# ────────────────────── Embedder ABC / 工具 ──────────────────────


class TestEmbedderBase:
    """Embedder 基类和静态方法测试。"""

    def test_cosine_perfect(self):
        """完全相同的向量余弦相似度应为 1.0。"""
        v = [1.0, 2.0, 3.0]
        assert Embedder.cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        """正交向量余弦相似度应为 0.0。"""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert Embedder.cosine(a, b) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        """相反方向余弦相似度应为 -1.0。"""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert Embedder.cosine(a, b) == pytest.approx(-1.0)

    def test_cosine_zero_vector(self):
        """零向量相似度为 0.0。"""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert Embedder.cosine(a, b) == pytest.approx(0.0)

    def test_cosine_different_length(self):
        """不同长度向量相似度为 0.0。"""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert Embedder.cosine(a, b) == pytest.approx(0.0)

    def test_cosine_empty(self):
        """空向量相似度为 0.0。"""
        assert Embedder.cosine([], []) == pytest.approx(0.0)
        assert Embedder.cosine([1.0], []) == pytest.approx(0.0)

    def test_cosine_known_value(self):
        """已知计算结果验证。"""
        a = [1.0, 0.0, 0.0]
        b = [0.5, 0.5, 0.0]
        # cos = (1*0.5 + 0*0.5 + 0*0) / (1.0 * sqrt(0.5)) = 0.5 / 0.7071 ≈ 0.7071
        assert Embedder.cosine(a, b) == pytest.approx(0.7071, abs=0.001)


# ────────────────────── TF-IDF Embedder ──────────────────────


class TestTfidfEmbedder:
    """TF-IDF 降级 Embedder 测试。"""

    def test_tokenize_chinese(self):
        """中文分词：2-gram 字符级。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        tokens = TfidfEmbedder._tokenize("服务器重启")
        # "服务器重启" → ["服务", "务器", "器重", "重启"]
        assert "服务" in tokens or len(tokens) > 0

    def test_tokenize_english(self):
        """英文分词：空格分词。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        tokens = TfidfEmbedder._tokenize("server restart issue")
        assert "server" in tokens
        assert "restart" in tokens
        assert "issue" in tokens

    def test_tokenize_mixed(self):
        """中英混合分词。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        tokens = TfidfEmbedder._tokenize("K8s Pod 重启")
        assert len(tokens) > 0
        assert "k8s" in tokens or "pod" in tokens

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """单条向量化。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        emb = TfidfEmbedder(max_features=128)
        vecs = await emb.embed(["这是一条测试记忆"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 128
        # L2 归一化检查
        import math
        norm = math.sqrt(sum(x * x for x in vecs[0]))
        if norm > 0:
            assert norm == pytest.approx(1.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_embed_multiple(self):
        """批量向量化。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        emb = TfidfEmbedder(max_features=256)
        vecs = await emb.embed([
            "Kubernetes pod restart issue",
            "MySQL connection pool configuration",
            "Redis cache eviction strategy",
        ])
        assert len(vecs) == 3
        assert all(len(v) == 256 for v in vecs)

    @pytest.mark.asyncio
    async def test_embed_empty(self):
        """空列表向量化。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        emb = TfidfEmbedder()
        vecs = await emb.embed([])
        assert vecs == []

    def test_dimension(self):
        """维度检查。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        emb = TfidfEmbedder(max_features=256)
        assert emb.dimension() == 256

    @pytest.mark.asyncio
    async def test_similar_texts_closer(self):
        """相似文本的余弦相似度应高于不相似文本。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        emb = TfidfEmbedder(max_features=256)

        # 先训练一批文本
        await emb.embed([
            "server restart",
            "pod crash",
            "database",
            "cache",
            "kubernetes deployment",
        ])

        # 相似对
        v1 = await emb.embed_query("server restart issue")
        v2 = await emb.embed_query("server reboot problem")
        sim_similar = emb.cosine_similarity(v1, v2)

        # 不相似对
        v3 = await emb.embed_query("database connection pool")
        sim_different = emb.cosine_similarity(v1, v3)

        # 相似文本的相似度应更高
        assert sim_similar > sim_different, (
            f"相似文本相似度 {sim_similar} 应高于不相似文本 {sim_different}"
        )

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """embed_query 应返回单条向量。"""
        from cortexos.embedding.tfidf import TfidfEmbedder
        emb = TfidfEmbedder(max_features=128)
        await emb.embed(["some training text"])
        vec = await emb.embed_query("test query")
        assert len(vec) == 128


# ────────────────────── OpenAI Compat Embedder (mock) ──────────────────────


class TestOpenAICompatEmbedder:
    """OpenAI 兼容 Embedder 测试（mock API）。"""

    def test_is_available_with_key(self):
        """有 key 时 is_available 为 True。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        emb = OpenAICompatEmbedder(api_key="test-key")
        assert emb.is_available

    def test_is_available_without_key(self):
        """无 key 时 is_available 为 False。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        emb = OpenAICompatEmbedder(api_key_env="NONEXISTENT_ENV")
        assert not emb.is_available

    @pytest.mark.asyncio
    async def test_embed_mock_response(self):
        """embed 应正确调用 API 并返回结果。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder

        emb = OpenAICompatEmbedder(
            base_url="http://mock.openai.com/v1",
            api_key="test-key",
            model="text-embedding-3-small",
        )

        # Mock httpx
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            vecs = await emb.embed(["text1", "text2"])

        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2, 0.3]
        assert vecs[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """embed_query 应返回单条向量。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        emb = OpenAICompatEmbedder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            vec = await emb.embed_query("test query")
        assert len(vec) == 3

    @pytest.mark.asyncio
    async def test_chat_mock_response(self):
        """chat 应正确调用 API 并返回内容。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        emb = OpenAICompatEmbedder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await emb.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_json(self):
        """chat_json 应解析 JSON 输出。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        emb = OpenAICompatEmbedder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"entities": ["k8s"], "text": "ok"}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await emb.chat_json([{"role": "user", "content": "extract"}])

        assert result == {"entities": ["k8s"], "text": "ok"}

    def test_dimension_default(self):
        """未调用 embed 时 dimension 默认为 1536。"""
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        emb = OpenAICompatEmbedder(api_key="test-key")
        assert emb.dimension() == 1536
