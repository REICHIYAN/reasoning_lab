from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    self_consistency_samples: int = 0

DEFAULT_SC_SAMPLES = 5
