import asyncio
import json
import logging
from json import JSONDecodeError
from typing import Dict, Generic, Iterable, List, Literal, Optional, TypeVar, Union, cast

import httpx
from openai import NOT_GIVEN, AsyncOpenAI, NotGiven, OpenAI
from openai._types import Body, Headers, Query
from openai.types import ChatModel, Metadata, ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionPredictionContentParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    completion_create_params,
)

from zipcoil.types import AsyncToolProtocol, ToolProtocol

ClientT = TypeVar("ClientT", OpenAI, AsyncOpenAI)
ToolT = TypeVar("ToolT", bound=Union[ToolProtocol, AsyncToolProtocol])


class BaseAgent(Generic[ClientT, ToolT]):
    def __init__(
        self,
        model: Union[str, ChatModel],
        client: ClientT,
        tools: Iterable[ToolT],
    ):
        self.model = model
        self.client = client
        self.tools: list[ToolT] = list(tools)  # Convert to list to ensure it's iterable multiple times
        self.tool_schemas: list[ChatCompletionToolParam] = []
        self.tool_map: dict[str, ToolT] = {}
        self.log = logging.getLogger(__name__)

        for tool_func in self.tools:
            try:
                schema = tool_func.tool_schema  # typed via the protocols
            except AttributeError as exc:
                raise ValueError(f"Tool {tool_func} is not decorated with @tool") from exc

            tool_name = schema["function"]["name"]
            if tool_name in self.tool_map:
                raise ValueError(f"Duplicate tool name: {tool_name}")

            self.tool_map[tool_name] = tool_func
            self.tool_schemas.append(schema)

    # _call_tool components
    def _prep_tool_call(self, name: str, args: dict) -> Optional[ToolT]:
        self.log.info(f"Calling tool `{name}` with `{args}`")
        return self.tool_map.get(name)

    def _tool_not_found(self, name: str) -> str:
        msg = f"Tool `{name}` not found"
        self.log.info(msg)
        return msg

    def _finalize_tool_result(self, name: str, result: object) -> str:
        result_str = result if isinstance(result, str) else str(result)
        self.log.info(f"Tool `{name}` returned `{result_str}`")
        return result_str

    def _finalize_tool_error(self, name: str, e: Exception) -> str:
        error_msg = f"Error executing tool `{name}`: `{str(e)}`"
        self.log.info(error_msg)
        return error_msg

    # run components
    def _run_prep_tool_name_and_args(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> tuple[str, Union[dict, JSONDecodeError]]:
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return name, e

        return name, args


class Agent(BaseAgent[OpenAI, ToolProtocol]):
    """An abstractization of the OpenAI tool-calling loop"""

    def __init__(
        self,
        model: Union[str, ChatModel],
        client: OpenAI,
        tools: Iterable[ToolProtocol],
    ) -> None:
        super().__init__(model, client, tools)

        for tool_func in self.tools:
            if asyncio.iscoroutinefunction(tool_func):
                tool_name = tool_func.tool_schema["function"]["name"]
                raise ValueError(
                    f"Tool `{tool_name}` is an async function, but this agent is synchronous. Please use AsyncAgent instead."
                )

    def _call_tool(self, name: str, args: dict) -> str:
        user_tool = self._prep_tool_call(name, args)
        if user_tool is None:
            return self._tool_not_found(name)

        try:
            result = user_tool(**args)
            return self._finalize_tool_result(name, result)
        except Exception as e:
            return self._finalize_tool_error(name, e)

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

            completion_choice = completion.choices[0]
            if completion_choice.finish_reason == "stop":
                return completion
            elif completion_choice.finish_reason != "tool_calls":
                raise ValueError(f"Unexpected finish reason: {completion.choices[0].finish_reason}")

            mutable_messages.append(cast(ChatCompletionMessageParam, completion_choice.message))
            assert completion_choice.message.tool_calls is not None  # type guard

            for tool_call in completion_choice.message.tool_calls:
                name, args_or_err = self._run_prep_tool_name_and_args(tool_call)
                if isinstance(args_or_err, JSONDecodeError):
                    mutable_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=f"Error decoding arguments for tool `{name}`: {str(args_or_err)}",
                        )
                    )
                    continue

                result = self._call_tool(name, cast(dict, args_or_err))
                mutable_messages.append(
                    ChatCompletionToolMessageParam(role="tool", tool_call_id=tool_call.id, content=result)
                )

        raise RuntimeError(f"Maximum iterations ({max_iterations}) reached without completion")


class AsyncAgent(BaseAgent[AsyncOpenAI, Union[ToolProtocol, AsyncToolProtocol]]):
    """An async abstractization of the OpenAI tool-calling loop.

    Allows using both sync and async tools.
    """

    def __init__(
        self,
        model: Union[str, ChatModel],
        client: AsyncOpenAI,
        tools: Iterable[Union[ToolProtocol, AsyncToolProtocol]],
    ) -> None:
        super().__init__(model, client, tools)

    async def _call_tool(self, name: str, args: dict) -> str:
        user_tool = self._prep_tool_call(name, args)
        if user_tool is None:
            return self._tool_not_found(name)

        try:
            if asyncio.iscoroutinefunction(user_tool):
                result = await user_tool(**args)
            else:
                result = user_tool(**args)
            return self._finalize_tool_result(name, result)
        except Exception as e:
            return self._finalize_tool_error(name, e)

    async def run(
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
            completion = await self.client.chat.completions.create(
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

            completion_choice = completion.choices[0]
            if completion_choice.finish_reason == "stop":
                return completion
            elif completion_choice.finish_reason != "tool_calls":
                raise ValueError(f"Unexpected finish reason: {completion.choices[0].finish_reason}")

            mutable_messages.append(cast(ChatCompletionMessageParam, completion_choice.message))
            assert completion_choice.message.tool_calls is not None  # type guard

            if completion.choices[0].finish_reason == "stop":
                return completion
            elif completion.choices[0].finish_reason == "tool_calls":
                message = completion.choices[0].message
                assert message.tool_calls is not None  # type guard

                for tool_call in completion_choice.message.tool_calls:
                    name, args_or_err = self._run_prep_tool_name_and_args(tool_call)
                    if isinstance(args_or_err, JSONDecodeError):
                        mutable_messages.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                tool_call_id=tool_call.id,
                                content=f"Error decoding arguments for tool `{name}`: {str(args_or_err)}",
                            )
                        )
                        continue

                    result = await self._call_tool(name, cast(dict, args_or_err))
                    mutable_messages.append(
                        ChatCompletionToolMessageParam(role="tool", tool_call_id=tool_call.id, content=result)
                    )
            else:
                raise ValueError(f"Unexpected finish reason: {completion.choices[0].finish_reason}")

        raise RuntimeError(f"Maximum iterations ({max_iterations}) reached without completion")
