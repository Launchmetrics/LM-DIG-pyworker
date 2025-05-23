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
class InputParameters:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 128

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputParameters":
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )


# @dataclasses.dataclass
# class InputParameters:
#    temperature: Optional[float] = None
#    max_tokens: Optional[int] = 128

#    @classmethod
#    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputParameters":
        # errors = {}
        # for param in inspect.signature(cls).parameters:
        #    if param not in json_msg:
        #        errors[param] = f"missing parameter (InputParameters): '{param}'"
        # if errors:
        #    raise JsonDataException(errors)
#        return cls(
#            **{
#                k: v
#                for k, v in json_msg.items()
#                if k in inspect.signature(cls).parameters
#            }
#        )


@dataclasses.dataclass
class InputData(ApiPayload):
    messages: list
    parameters: InputParameters
    max_tokens: int = 16

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
            parameters=InputParameters(max_tokens=256),
            max_tokens=256
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return self.parameters.max_tokens

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = f"missing parameter (InputData): '{param}'"
        if errors:
            raise JsonDataException(errors)
        try:
            parameters = InputParameters.from_json_msg(json_msg["parameters"])
            return cls(
                messages=json_msg["messages"],
                parameters=parameters,
                max_tokens=parameters.max_tokens
            )
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)
