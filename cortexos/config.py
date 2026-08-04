"""配置管理模块 —— 全参数化 YAML + 环境变量覆盖。

所有阈值、权重、天数、路径均通过配置文件管理，
支持环境变量 `CORTEXOS_*` 覆盖任意嵌套键。
"""

import os
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional, get_type_hints

import yaml


def _env_override(cfg: Dict[str, Any], prefix: str = "CORTEXOS_") -> Dict[str, Any]:
    """环境变量覆盖配置。

    环境变量命名规则：
      server.host                 → CORTEXOS_SERVER_HOST
      zone.emergence.semantic_threshold → CORTEXOS_ZONE_EMERGENCE_SEMANTIC_THRESHOLD
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        rest = env_key[len(prefix):].lower()
        parts = rest.split("_")

        # 智能路径解析：从最浅到最深尝试匹配 dict 层级
        def _set_nested(d: Dict[str, Any], path_parts: List[str]) -> None:
            """尝试用 path_parts 的连续前缀匹配 dict 层级。"""
            cursor = d
            i = 0
            while i < len(path_parts):
                # 尝试找最长匹配的 key（贪心优先匹配更长的组合键）
                found = False
                end_range = len(path_parts)
                # 最后一级是叶子，不需要子 dict 匹配
                search_end = len(path_parts) if i == len(path_parts) - 1 else len(path_parts) - 1
                for j in range(len(path_parts), i, -1):
                    candidate = "_".join(path_parts[i:j])
                    if candidate in cursor:
                        if j == len(path_parts):
                            # 剩余全部匹配→叶子节点，直接设置
                            val: Any = env_val
                            try:
                                val = int(val)
                            except ValueError:
                                try:
                                    val = float(val)
                                except ValueError:
                                    if val.lower() in ("true", "false"):
                                        val = val.lower() == "true"
                            cursor[candidate] = val
                            return
                        elif isinstance(cursor[candidate], dict):
                            cursor = cursor[candidate]
                            i = j
                            found = True
                            break
                if not found:
                    return

        _set_nested(cfg, parts)
    return cfg


def _dc_from_dict(dc_cls: type, raw: Dict[str, Any]) -> Any:
    """递归地从字典构建 dataclass。"""
    hints = get_type_hints(dc_cls)
    kwargs: Dict[str, Any] = {}
    for f in dc_fields(dc_cls):
        if f.name in raw:
            raw_val = raw[f.name]
            field_type = hints.get(f.name)
            if isinstance(raw_val, dict) and _is_dataclass_type(field_type):
                kwargs[f.name] = _dc_from_dict(field_type, raw_val)
            else:
                kwargs[f.name] = raw_val
        else:
            # 使用默认值
            from dataclasses import MISSING
            if f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            elif f.default is not MISSING:
                kwargs[f.name] = f.default
    return dc_cls(**kwargs)


def _is_dataclass_type(t: Optional[type]) -> bool:
    """判断类型是否为 dataclass。"""
    if t is None:
        return False
    if hasattr(t, "__dataclass_fields__"):
        return True
    # 处理 Optional[SomeDC] 等泛型
    origin = getattr(t, "__origin__", None)
    if origin is not None:
        args = getattr(t, "__args__", ())
        for arg in args:
            if arg is not type(None) and hasattr(arg, "__dataclass_fields__"):
                return True
    return False


# ────────────────────── 子配置 ──────────────────────


@dataclass
class ServerConfig:
    """服务器配置。"""
    host: str = "0.0.0.0"
    port: int = 8200
    tls: bool = False


@dataclass
class LocalStorageConfig:
    """本地 SQLite 存储配置。"""
    path: str = "./data/memory.db"
    jsonl_export: str = "./data/export"


@dataclass
class ObjectStorageConfig:
    """对象存储配置（S3 兼容）。"""
    bucket: str = "agent-cortexos-memory"
    endpoint: str = ""
    region: str = "us-east-1"
    index_db: str = "./data/index.db"


@dataclass
class StorageConfig:
    """存储总配置。"""
    backend: str = "local"
    local: LocalStorageConfig = field(default_factory=LocalStorageConfig)
    object: ObjectStorageConfig = field(default_factory=ObjectStorageConfig)


@dataclass
class LLMConfig:
    """LLM 配置（OpenAI 兼容）。"""
    provider: str = "openai_compat"
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    model: str = ""
    embedding_model: str = ""
    extract_entities: bool = True
    extract_facts: bool = False
    extract_edges: bool = False


@dataclass
class ZoneEmergenceConfig:
    """Zone 涌现配置。"""
    base_threshold: int = 5
    semantic_threshold: float = 0.72
    min_content_len: int = 20
    cluster_similarity: float = 0.75


@dataclass
class ZoneLifecycleConfig:
    """Zone 生命周期配置。"""
    dormant_days: int = 30
    archive_days: int = 90
    merge_threshold: float = 0.7


@dataclass
class ZoneGravityConfig:
    """Zone 重力公式参数。"""
    decay_lambda: float = 0.02
    activity_k: float = 50.0
    scale_m: float = 100.0


@dataclass
class ZoneConfig:
    """Zone 总配置。"""
    emergence: ZoneEmergenceConfig = field(default_factory=ZoneEmergenceConfig)
    lifecycle: ZoneLifecycleConfig = field(default_factory=ZoneLifecycleConfig)
    gravity: ZoneGravityConfig = field(default_factory=ZoneGravityConfig)


@dataclass
class RecallWeights:
    """多因子评分权重。"""
    text_sim: float = 0.35
    recency: float = 0.25
    gravity: float = 0.15
    freq: float = 0.10
    scope_boost: float = 0.05
    graph_path: float = 0.10


@dataclass
class RecallConfig:
    """检索配置。"""
    weights: RecallWeights = field(default_factory=RecallWeights)
    rrf_k: int = 60
    recency_half_life_days: float = 7.0
    graph_hop: int = 2
    graph_decay: float = 0.5
    top_k: int = 20


@dataclass
class ResolveConfig:
    """冲突消解配置。"""
    enabled: bool = True
    conflict_window_days: int = 30


@dataclass
class ConsolidateConfig:
    """整合引擎配置。"""
    time_gate_hours: int = 24
    content_gate_count: int = 50
    raw_summary_threshold: int = 3
    similarity_threshold: float = 0.8
    index_budget_count: int = 200
    index_budget_bytes: int = 25600


@dataclass
class RateLimitConfig:
    """限流配置。"""
    default: int = 100
    pair_request: int = 10


@dataclass
class PairConfig:
    """配对配置。"""
    code_length: int = 8
    code_expire_minutes: int = 15
    key_rotation_days: int = 90


# ────────────────────── 顶层配置 ──────────────────────


@dataclass
class Config:
    """CortexOS 顶层配置 —— 所有参数化入口。

    加载优先级：默认值 → YAML 文件 → 环境变量 CORTEXOS_*
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    resolve: ResolveConfig = field(default_factory=ResolveConfig)
    consolidate: ConsolidateConfig = field(default_factory=ConsolidateConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    pair: PairConfig = field(default_factory=PairConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置，再应用环境变量覆盖。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data = _env_override(data)
        return _dc_from_dict(Config, data)

    @classmethod
    def from_env(cls) -> "Config":
        """从纯环境变量加载配置（默认值 + 环境变量覆盖）。"""
        base = Config().to_dict()
        data = _env_override(base)
        return _dc_from_dict(Config, data)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """智能加载配置。"""
        if config_path and os.path.isfile(config_path):
            return cls.from_yaml(config_path)
        return cls.from_env()

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典。"""
        return _to_dict_recursive(self)


def _to_dict_recursive(obj: Any) -> Any:
    """递归将 dataclass 转字典。"""
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_dict_recursive(getattr(obj, f.name)) for f in dc_fields(obj)}
    return obj


def load_config(path: Optional[str] = None) -> Config:
    """便捷函数：加载配置。"""
    return Config.load(config_path=path)
