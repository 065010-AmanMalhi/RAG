"""
config.py — Parse models.txt and expose all configuration as a dataclass.
Supports parameterized multi-instance runs.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration parsed from models.txt."""

    # LLM
    llm_name: str = "lukaspetrik/gemma3-tools:1b"
    llm_provider: str = "ollama"
    llm_size: str = "983MB"
    llm_context_window: str = "4K"

    # Embedding
    embedding_model_name: str = "jina/jina-embeddings-v2-small-en"
    embedding_model_provider: str = "ollama"
    embedding_model_size: str = "96MB"
    embedding_model_context_window: str = "8K"

    # Judge LLM
    judge_llm: str = "llama3.2:3b"
    judge_llm_provider: str = "ollama"

    # Evaluation
    evaluation_framework: str = "ragas-llamaindex"

    # Database
    db_host: str = "192.240.1.177"
    db_name: str = "bhavya_manya"
    db_role: str = "bhavya_manya"
    db_user: str = "bhavya_manya"
    db_password: str = "bhavya_manya"
    db_vector_extension: str = "pgvector"

    # File paths
    dataset_path: str = "/home/ashok/RAGSYSTEMS/rag_system_llama_3 (aman)/dataset1.txt"
    questions_path: str = "/home/ashok/RAGSYSTEMS/rag_system_llama_3 (aman)/Questions.txt"
    answers_path: str = "/home/ashok/RAGSYSTEMS/rag_system_llama_3 (aman)/Answers.txt"

    # Output
    output_file: str = "output_rag.json"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Vector store
    collection_name: str = "rag_documents"

    @property
    def pg_connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:5432/{self.db_name}"
        )

    @classmethod
    def from_models_file(cls, filepath: str = "models.txt") -> "Config":
        """Parse a models.txt config file and return a Config instance."""
        config_dict = {}
        path = Path(filepath)
        if not path.exists():
            print(f"[Config] Warning: {filepath} not found, using defaults.")
            return cls()

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    config_dict[key.strip()] = value.strip()

        # Map config file keys to dataclass fields
        key_map = {
            "LLM_NAME": "llm_name",
            "LLM_PROVIDER": "llm_provider",
            "LLM_SIZE": "llm_size",
            "LLM_CONTEXT_WINDOW": "llm_context_window",
            "EMBEDDING_MODEL_NAME": "embedding_model_name",
            "EMBEDDING_MODEL_PROVIDER": "embedding_model_provider",
            "EMBEDDING_MODEL_SIZE": "embedding_model_size",
            "EMBEDDING_MODEL_CONTEXT_WINDOW": "embedding_model_context_window",
            "JUDGE_LLM": "judge_llm",
            "JUDGE_LLM_PROVIDER": "judge_llm_provider",
            "EVALUATION_FRAMEWORK": "evaluation_framework",
            "DB_HOST": "db_host",
            "DB_NAME": "db_name",
            "DB_ROLE": "db_role",
            "DB_USER": "db_user",
            "DB_VECTOR_EXTENSION": "db_vector_extension",
            "DATASET_PATH": "dataset_path",
            "QUESTIONS_PATH": "questions_path",
            "ANSWERS_PATH": "answers_path",
            "OUTPUT_FILE": "output_file",
        }

        kwargs = {}
        for file_key, attr_name in key_map.items():
            if file_key in config_dict:
                kwargs[attr_name] = config_dict[file_key]

        # Map config file keys to dataclass fields

        return cls(**kwargs)


def load_config(models_file: str = None) -> Config:
    """Load configuration from models.txt or defaults."""
    if models_file is None:
        # Look in current directory or script directory
        script_dir = Path(__file__).parent
        models_file = str(script_dir / "models.txt")
    return Config.from_models_file(models_file)


if __name__ == "__main__":
    cfg = load_config()
    print(f"LLM: {cfg.llm_name}")
    print(f"Embedding: {cfg.embedding_model_name}")
    print(f"DB: {cfg.pg_connection_string}")
    print(f"Eval Framework: {cfg.evaluation_framework}")
    print(f"Judge: {cfg.judge_llm}")
