from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class ExperimentConfig:
    exp_name: str = "default"
    memory_type: str = "none"  # "none", "dilu", "global_rag"
    use_reflection: bool = False
    task_count: int = 10
    model_name: str = "gemini-2.5-flash"
    
    # Paths
    base_dir: str = field(default_factory=lambda: "path/to/root")
    data_dir: str = "dataset"
    task_set: str = "amazon"
    
    # RAG Settings
    global_db_path: str = "global_chroma_db"
    
    def to_dict(self):
        return asdict(self)