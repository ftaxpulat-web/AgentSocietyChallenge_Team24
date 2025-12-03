# team24_agent/config.py

from dataclasses import dataclass, asdict, field
import os

@dataclass
class ExperimentConfig:
    exp_name: str = "default"

    # module choices (must match registry keys)
    memory_type: str = "none"      # "none", "dilu", "generative", "tp", "voyager"
    reasoning_type: str = "io"     # "io", "cot", "cotsc", "tot", "dilu", "self_refine", "step_back"
    planning_type: str = "io"      # "io", "deps", "td", "voyager", "openagi", "hugginggpt"

    task_count: int = 10
    model_name: str = "gemini-2.5-flash"

    base_dir: str = field(default_factory=lambda: os.path.dirname(__file__))
    data_dir: str = "dataset"

    def to_dict(self):
        return asdict(self)
