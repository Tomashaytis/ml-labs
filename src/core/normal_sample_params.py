import numpy as np

from typing import Dict, Any
from dataclasses import dataclass

from src.core import SampleParams


@dataclass
class NormalSampleParams(SampleParams):
    mean: np.ndarray
    covariance: np.ndarray

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result['mean'] = self.mean.tolist()
        result['covariance'] = self.covariance.tolist()
        result['__class__'] = 'NormalSampleParams'
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'NormalSampleParams':
        mean = np.array(data['mean'])
        covariance = np.array(data['covariance'])
        params_data = {k: v for k, v in data.items()
                       if k not in ['mean', 'covariance', '__class__']}
        return cls(mean=mean, covariance=covariance, **params_data)