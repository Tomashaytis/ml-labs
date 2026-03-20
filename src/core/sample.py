import numpy as np

from typing import Union, Dict, Any
from dataclasses import dataclass

from src.core import SampleParams, NormalSampleParams


@dataclass
class Sample:
    data: np.ndarray
    params: Union[SampleParams, NormalSampleParams]

    def to_json(self) -> Dict[str, Any]:
        result = {
            'data': self.data.tolist(),
            'params': self.params.to_json(),
            '__class__': 'Sample'
        }
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Sample':
        ndarray_data = np.array(data['data'])
        params_data = data['params']
        params_class_name = params_data.get('__class__', 'SampleParams')

        if params_class_name == 'NormalSampleParams':
            params = NormalSampleParams.from_json(params_data)
        else:
            params = SampleParams.from_json(params_data)

        return cls(data=ndarray_data, params=params)
