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
    """
    Supports almost all params from /v1/chat/completions
    - https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/chat_completions
    """
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    logprobs: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    model: Optional[str] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[str] = None
    seed: Optional[int] = None
    stream: Optional[bool] = None
    stream_options: Optional[Any] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None

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
    max_tokens: Optional[int] = None

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
                max_tokens=parameters.max_tokens
            )
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)
