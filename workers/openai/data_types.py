import dataclasses
from typing import Dict, Any, Optional
from lib.data_types import ApiPayload, JsonDataException
from tasks.brand import bench_messages


def no_default_str(cls):
    # Decorator for class
    def __str__(self):
        """Returns a string containing only the non-default field values."""
        name_value = ', '.join(
            f'{field.name}={getattr(self, field.name)}'
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != (
                field.default
                if field.default is not dataclasses.MISSING
                else field.default_factory()
                if field.default_factory is not dataclasses.MISSING
                else dataclasses.MISSING
            )
        )
        return f'{type(self).__name__}({name_value})'

    setattr(cls, '__str__', __str__)
    return cls


# https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams

@dataclasses.dataclass
@no_default_str
class InputData(ApiPayload):
    messages: list  # mandadatory

    req_id: Optional[str] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.max_tokens
    max_tokens: Optional[int] = None

    max_completion_tokens: Optional[int] = None

    reasoning_effort: Optional[str] = None

    response_format: Optional[dict] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.temperature
    temperature: Optional[float] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.logprobs
    logprobs: Optional[int] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.n
    n: Optional[int] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.presence_penalty
    presence_penalty: Optional[float] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.repetition_penalty
    repetition_penalty: Optional[float] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.seed
    seed: Optional[int] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.stop
    stop: Optional[list[str]] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.top_k
    top_k: Optional[int] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.top_p
    top_p: Optional[float] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.min_p
    min_p: Optional[float] = None

    # https://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.logit_bias
    logit_bias: Optional[dict[int, float]] = None

    # fhttps://docs.vllm.ai/en/stable/api/vllm/#vllm.SamplingParams.frequency_penalty
    frequency_penalty: Optional[float] = None

    @classmethod
    def for_test(cls) -> "InputData":
        return cls(
            messages=bench_messages,
            max_tokens=2,
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return self.max_tokens

    @classmethod
    def from_json_msg(cls, batch: list[Dict[str, Any]]) -> list["InputData"]:
        errors = {}
        for json_msg in batch:
            for field in dataclasses.fields(cls):
                if type(field.default) == dataclasses._MISSING_TYPE and json_msg.get(field.name) is None:
                    errors[field.name] = f"missing parameter (InputData): '{field.name}'"
            if errors:
                raise JsonDataException(errors)
        try:
            return [cls(**json_msg) for json_msg in batch]
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)
