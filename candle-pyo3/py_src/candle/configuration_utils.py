from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import inspect


@dataclass
class PretrainedConfig(ABC):
    model_type: str = ""

    @classmethod
    def _parameters(cls) -> List[str]:
        return [e for e in cls.__dict__.keys() if not e.startswith("_") and not inspect.ismethod(getattr(cls, e))]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PretrainedConfig":
        params = {k: v for k, v in config_dict.items() if k in cls._parameters()}
        instance = cls()
        for k, v in params.items():
            setattr(instance, k, v)
        return instance

    def to_dict(self, remove_none: bool = False) -> Dict[str, Any]:
        config = {}
        for field in fields(self):
            if remove_none and getattr(self, field.name) is None:
                continue
            config[field.name] = getattr(self, field.name)
        return config
