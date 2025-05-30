import dataclasses
import inspect
from typing import Dict, Any, Optional
from lib.data_types import ApiPayload, JsonDataException


def no_default_str(cls):  # Decorator for class.
    def __str__(self):
        """Returns a string containing only the non-default field values."""
        s = ', '.join(f'{field.name}={getattr(self, field.name)}'
                      for field in dataclasses.fields(self)
                      if getattr(self, field.name) != field.default)
        return f'{type(self).__name__}({s})'

    setattr(cls, '__str__', __str__)
    return cls


@dataclasses.dataclass
@no_default_str
class InputData(ApiPayload):
    messages: list
    max_tokens: int
    temperature: Optional[float] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None
    seed: Optional[int] = None
    top_p: Optional[float] = None

    # logprobs: Optional[bool] = None
    # frequency_penalty: Optional[float] = None
    # model: Optional[str] = None
    # presence_penalty: Optional[float] = None
    # response_format: Optional[str] = None
    # stream: Optional[bool] = None
    # stream_options: Optional[Any] = None
    # top_logprobs: Optional[int] = None

    @classmethod
    def for_test(cls) -> "InputData":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is deep learning?"
            }
        ]
        return cls(
            messages=messages,
            max_tokens=256,
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return self.max_tokens

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputData":
        errors = {}
        for field in dataclasses.fields(cls):
            if json_msg.get(field.name) == field.default is not None:
                errors[field.name] = f"missing parameter (InputData): '{field.name}'"
        if errors:
            raise JsonDataException(errors)
        try:
            return cls(**json_msg)
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)
