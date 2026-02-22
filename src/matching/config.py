"""Configuration — weights, thresholds, model parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DimensionWeights(BaseModel):
    complementarity: float = Field(default=0.50, ge=0.0, le=1.0)
    transaction_readiness: float = Field(default=0.30, ge=0.0, le=1.0)
    non_obvious: float = Field(default=0.20, ge=0.0, le=1.0)


class ComplementarityWeights(BaseModel):
    needs_provides_alignment: float = 0.50
    value_chain_adjacency: float = 0.30
    bidirectional_multiplier: float = 0.20


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_fast_model: str = "claude-3-haiku-20240307"

    dimension_weights: DimensionWeights = DimensionWeights()
    complementarity_weights: ComplementarityWeights = ComplementarityWeights()

    bidirectional_threshold: float = 0.3
    bidirectional_boost: float = 1.3

    top_k: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
