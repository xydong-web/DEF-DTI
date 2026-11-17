"""Public DEF-DTI package with the original SCR-DTI API for compatibility."""

from scrdti.config import ModelConfig, ReleaseConfig, TrainConfig, load_config
from scrdti.model import SCRDTI, SCRDTIOutput

__version__ = "0.2.0"

__all__ = [
    "ModelConfig",
    "ReleaseConfig",
    "SCRDTI",
    "SCRDTIOutput",
    "TrainConfig",
    "load_config",
]

