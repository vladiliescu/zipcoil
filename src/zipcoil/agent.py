import asyncio
import json
import logging
from typing import Dict, Iterable, List, Literal, Optional, Union

import httpx
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai._types import Body, Headers, Query
from openai.types import ChatModel, Metadata, ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    completion_create_params,
)

from zipcoil.types import ToolProtocol


class Agent:
    """An abstractization of the OpenAI tool-calling loop"""

    def __init__(
        self,
        model: Union[str, ChatModel],
        client: OpenAI,
        tools: Iterable[ToolProtocol],
    ) -> None:
        self.model = model
        self.client = client
        self.tools = list(tools)  # Convert to list to ensure it's iterable multiple times
        self.tool_schemas: list[dict[str, object]] = []
        self.tool_map: dict[str, ToolProtocol] = {}
        self.log = logging.getLogger(__name__)

        for tool_func in self.tools:
            if not hasattr(tool_func, "_tool_schema"):
                raise ValueError(f"Tool {tool_func} is not decorated with @tool")

            tool_name = tool_func._tool_schema["function"]["name"]
            if tool_name in self.tool_map:
                raise ValueError(f"Duplicate tool name: {tool_name}")

            if asyncio.iscoroutinefunction(tool_func):
                raise ValueError(
                    f"Tool `{tool_name}` is an async function, but this agent is synchronous. Please use AsyncAgent instead."
                )

            self.tool_map[tool_name] = tool_func
            self.tool_schemas.append(tool_func._tool_schema)

    def _call_tool(self, name: str, args: dict):
        self.log.info(f"Calling tool `{name}` with `{args}`")

        user_tool = self.tool_map.get(name, None)
        if user_tool is None:
            return f"Tool `{name}` not found"

        try:
            result = user_tool(**args)
            result_str = str(result) if not isinstance(result, str) else result

            self.log.info(f"Tool `{name}` returned `{result_str}`")
            return result_str
        except Exception as e:
            error_msg = f"Error executing tool `{name}`: `{str(e)}`"

            self.log.info(error_msg)
            return error_msg

    def run(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        audio: Optional[ChatCompletionAudioParam] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        modalities: Optional[List[Literal["text", "audio"]]] | NotGiven = NOT_GIVEN,
        # n: Optional[int] | NotGiven = NOT_GIVEN,
        # parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        prediction: Optional[ChatCompletionPredictionContentParam] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[ReasoningEffort] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        # stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        # stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        # tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        # tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_iterations: int = 10,
    ) -> ChatCompletion:
        mutable_messages = list(messages)
        for iteration in range(max_iterations):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=mutable_messages,
                tools=self.tool_schemas,
                n=1,  # Only one completion at a time otherwise the logic gets messy
                audio=audio,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                metadata=metadata,
                modalities=modalities,
                prediction=prediction,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            if completion.choices[0].finish_reason == "stop":
                return completion
            elif completion.choices[0].finish_reason == "tool_calls":
                mutable_messages.append(completion.choices[0].message)

                assert completion.choices[0].message.tool_calls is not None  # type guard
                for tool_call in completion.choices[0].message.tool_calls:
                    name = tool_call.function.name

                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        mutable_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error decoding arguments for tool `{name}`: {str(e)}",
                            }
                        )
                        continue

                    result = self._call_tool(name, args)
                    mutable_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            else:
                raise ValueError(f"Unexpected finish reason: {completion.choices[0].finish_reason}")

        raise RuntimeError(f"Maximum iterations ({max_iterations}) reached without completion")
