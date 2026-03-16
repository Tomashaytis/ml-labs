from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class SampleParams:
    prior: float
    class_label: int
    dimensional: int
    class_name: Optional[str]
    class_color: Optional[str]

    def to_json(self) -> Dict[str, Any]:
        result = asdict(self)
        result = {k: v for k, v in result.items() if v is not None}
        result['__class__'] = 'SampleParams'
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SampleParams':
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        return cls(**data)