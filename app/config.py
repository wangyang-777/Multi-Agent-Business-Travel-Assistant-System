from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用与 Agent 运行时配置（自环境变量与 `.env` 加载）。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "商旅-agent-guide"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_max_tokens: int = 8192
    embedding_chunk_max_tokens: int = 7000
    travel_inventory_provider: str = "demo"
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
    amadeus_base_url: str = "https://test.api.amadeus.com"
    flyai_cli_bin: str = "flyai"
    flyai_api_key: str = ""
    flyai_timeout_s: float = 45.0
    travel_search_max_results: int = 5
    railway_12306_skill_dir: str = ""
    railway_12306_node_bin: str = "node"
    railway_mcp_url: str = ""
    railway_mcp_timeout_s: float = 30.0
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/travel_agent"
    redis_url: str = "redis://localhost:6379/0"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    log_level: str = "INFO"

    # Agent config
    agent_orchestrator_backend: str = "langgraph"
    max_react_iterations: int = 10
    memory_window_size: int = 20
    memory_summary_threshold: int = 15
    memory_max_messages: int = 40
    memory_session_ttl_seconds: int = 86400

    # Circuit breaker config
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30
    circuit_breaker_half_open_max_calls: int = 3


settings = Settings()
