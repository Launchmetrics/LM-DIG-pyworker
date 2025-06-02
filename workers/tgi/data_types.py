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
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[list[float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    stream: bool = False
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[list] = None
    tool_prompt: Optional[str] = None
    tool_choice: Optional[str] = None
    stop: Optional[list[str]] = None

    """
        Given a list of messages, generate a response.
        - https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
        Check ChatRequest:
        - https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/types.py

        Args:
            messages (`List`):
                List of messages (dict) in the conversation
            repetition_penalty (`float`):
                The parameter for repetition penalty. 0.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            frequency_penalty (`float`):
                The parameter for frequency penalty. 0.0 means no penalty
                Penalize new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            logit_bias (`List[float]`):
                Adjust the likelihood of specified tokens
            logprobs (`bool`):
                Include log probabilities in the response
            top_logprobs (`int`):
                Include the `n` most likely tokens at each step
            max_tokens (`int`):
                Maximum number of generated tokens
            n (`int`):
                Generate `n` completions
            presence_penalty (`float`):
                The parameter for presence penalty. 0.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            stream (`bool`):
                Stream the response
            seed (`int`):
                Random sampling seed
            temperature (`float`):
                The value used to module the logits distribution.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation
            tools (`List[Tool]`):
                List of tools to use
            tool_prompt (`str`):
                A prompt to be appended before the tools
            tool_choice (`str`):
                The tool to use
            stop (`List[str]`):
                Stop generating tokens if a member of `stop` is generated

    """

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
            if field.default is not None and json_msg.get(field.name) is None:
                errors[field.name] = f"missing parameter (InputData): '{field.name}'"
        if errors:
            raise JsonDataException(errors)
        try:
            return cls(**json_msg)
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)
