from typing import Optional, Dict, Any, List

from phi.llm.message import Message
from phi.llm.aws.bedrock import AwsBedrock


class Claude(AwsBedrock):
    def __init__(self,
                 name: str = "AwsBedrockAnthropicClaude",
                 model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
                 max_tokens: int = 8192,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 stop_sequences: Optional[List[str]] = None,
                 anthropic_version: str = "bedrock-2023-05-31",
                 request_params: Optional[Dict[str, Any]] = None,
                 client_params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self.anthropic_version = anthropic_version
        self.request_params = request_params
        self.client_params = client_params

        # Validate the parameters
        self.check_model_id()
        self.check_hyperparameters()

    def check_model_id(self):
        ALLOWED_CLAUDE_IDS = ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-opus-20240229-v1:0",
                              "anthropic.claude-3-haiku-20240307-v1:0"]
        assert self.model in ALLOWED_CLAUDE_IDS, f"When using AWS Claude messaging API, you should choose from this list: {ALLOWED_CLAUDE_IDS}\nYou passed {self.model}"

    def check_hyperparameters(self):
        if self.temperature:
            assert 0 <= self.temperature <= 1, "temperature must be between 0 and 1."
        if self.top_p:
            assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1."
        if self.top_k:
            assert 0 <= self.top_k <= 500, "top_k must be between 0 and 500."

    def to_dict(self) -> Dict[str, Any]:
        _dict = super().to_dict()
        _dict["max_tokens"] = self.max_tokens
        _dict["temperature"] = self.temperature
        _dict["top_p"] = self.top_p
        _dict["top_k"] = self.top_k
        _dict["stop_sequences"] = self.stop_sequences
        return _dict

    @property
    def api_kwargs(self) -> Dict[str, Any]:
        _request_params: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "anthropic_version": self.anthropic_version,
        }
        if self.temperature:
            _request_params["temperature"] = self.temperature
        if self.top_p:
            _request_params["top_p"] = self.top_p
        if self.top_k:
            _request_params["top_k"] = self.top_k
        if self.stop_sequences:
            _request_params["stop_sequences"] = self.stop_sequences
        if self.request_params:
            _request_params.update(self.request_params)
        return _request_params

    def get_request_body(self, messages: List[Message]) -> Dict[str, Any]:
        system_prompt = None
        messages_for_api = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                messages_for_api.append({"role": m.role, "content": m.content})

        # -*- Build request body
        request_body = {
            "messages": messages_for_api,
            **self.api_kwargs,
        }
        if system_prompt:
            request_body["system"] = system_prompt
        return request_body

    def parse_response_message(self, response: Dict[str, Any]) -> Message:
        if response.get("type") == "message":
            response_message = Message(role=response.get("role"))
            content: Optional[str] = ""
            if response.get("content"):
                _content = response.get("content")
                if isinstance(_content, str):
                    content = _content
                elif isinstance(_content, dict):
                    content = _content.get("text", "")
                elif isinstance(_content, list):
                    content = "\n".join([c.get("text") for c in _content])

            response_message.content = content
            return response_message

        return Message(
            role="assistant",
            content=response.get("completion"),
        )

    def parse_response_delta(self, response: Dict[str, Any]) -> Optional[str]:
        if "delta" in response:
            return response.get("delta", {}).get("text")
        return response.get("completion")
