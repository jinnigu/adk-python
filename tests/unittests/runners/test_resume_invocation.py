# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for edge cases of resuming invocations."""

import asyncio
import copy
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
from google.genai.types import FunctionResponse
from google.genai.types import Part
import pytest

from .. import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name="transfer_to_agent", args={"agent_name": agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name="transfer_to_agent", response={"result": None}
)


def test_tool() -> dict[str, str]:
  return {"result": "test tool result"}


test_tool.__test__ = False


class _ParallelEscalationTestingAgent(BaseAgent):
  """A testing agent that emits a single event after a delay."""

  delay: float = 0
  response_text: str = ""
  escalate: bool = False
  emit_follow_up_after_first_event: bool = False

  def _create_event(
      self,
      ctx: InvocationContext,
      text: str,
      *,
      escalate: bool = False,
  ) -> Event:
    return Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(escalate=True) if escalate else EventActions(),
    )

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    await asyncio.sleep(self.delay)
    yield self._create_event(ctx, self.response_text, escalate=self.escalate)
    if self.emit_follow_up_after_first_event:
      yield self._create_event(ctx, "This event should not be emitted.")


@pytest.mark.asyncio
async def test_resume_invocation_from_sub_agent():
  """A test case for an edge case, where an invocation-to-resume starts from a sub-agent.

  For example:
    invocation1: root_agent -> sub_agent
    invocation2: sub_agent [paused][resume]
  """
  # Step 1: Setup
  # root_agent -> sub_agent
  sub_agent = LlmAgent(
      name="sub_agent",
      model=testing_utils.MockModel.create(
          responses=[
              "first response from sub_agent",
              "second response from sub_agent",
              "third response from sub_agent",
          ]
      ),
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[transfer_call_part(sub_agent.name)]
      ),
      sub_agents=[sub_agent],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Step 2: Run the first invocation
  # Expect the invocation to start from root_agent and transferred to sub_agent.
  invocation_1_events = await runner.run_async("test user query")
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_1_events)
  ) == [
      (
          root_agent.name,
          transfer_call_part(sub_agent.name),
      ),
      (
          root_agent.name,
          TRANSFER_RESPONSE_PART,
      ),
      (
          sub_agent.name,
          "first response from sub_agent",
      ),
      (
          sub_agent.name,
          testing_utils.END_OF_AGENT,
      ),
      (
          root_agent.name,
          testing_utils.END_OF_AGENT,
      ),
  ]

  # Step 3: Run the second invocation
  # Expect the invocation to directly start from sub_agent.
  invocation_2_events = await runner.run_async(
      "test user query 2",
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_2_events)
  ) == [
      (
          sub_agent.name,
          "second response from sub_agent",
      ),
      (sub_agent.name, testing_utils.END_OF_AGENT),
  ]
  # Asserts the invocation will be a no-op if the current agent in context is
  # already final.
  assert not await runner.run_async(
      invocation_id=invocation_2_events[0].invocation_id
  )

  # Step 4: Copy all session.events[:-1] to a new session
  # This is to simulate the case where we pause on the second invocation.
  session_id = runner.session_id
  session = await runner.runner.session_service.get_session(
      app_name="test_app", user_id="test_user", session_id=session_id
  )
  new_session = await runner.runner.session_service.create_session(
      app_name=session.app_name, user_id=session.user_id
  )
  for event in session.events[:-1]:
    await runner.runner.session_service.append_event(new_session, event)
  runner.session_id = new_session.id

  # Step 5: Resume the second invocation
  resumed_invocation_2_events = await runner.run_async(
      invocation_id=invocation_2_events[0].invocation_id
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(resumed_invocation_2_events)
  ) == [
      (
          sub_agent.name,
          "third response from sub_agent",
      ),
      (sub_agent.name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resume_any_invocation():
  """A test case for resuming a previous invocation instead of the last one."""
  # Step 1: Setup
  long_running_test_tool = LongRunningFunctionTool(
      func=test_tool,
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[
              Part.from_function_call(name="test_tool", args={}),
              "llm response in invocation 2",
              Part.from_function_call(name="test_tool", args={}),
              "llm response after resuming invocation 1",
          ]
      ),
      tools=[long_running_test_tool],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Step 2: Run the first invocation, which pauses on the long running function.
  invocation_1_events = await runner.run_async("test user query")
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_1_events)
  ) == [
      (
          root_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
      (
          root_agent.name,
          Part.from_function_response(
              name="test_tool", response={"result": "test tool result"}
          ),
      ),
  ]

  # Step 3: Run the second invocation, expect it to finish normally.
  invocation_2_events = await runner.run_async(
      "test user query 2",
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_2_events)
  ) == [
      (
          root_agent.name,
          "llm response in invocation 2",
      ),
      (root_agent.name, testing_utils.END_OF_AGENT),
  ]

  # Step 4: Run the third invocation, which also pauses on the long running
  # function.
  invocation_3_events = await runner.run_async(
      "test user query 3",
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_3_events)
  ) == [
      (
          root_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
      (
          root_agent.name,
          Part.from_function_response(
              name="test_tool", response={"result": "test tool result"}
          ),
      ),
  ]

  # Step 5: Resume the first invocation with long running function response.
  resumed_invocation_1_events = await runner.run_async(
      invocation_id=invocation_1_events[0].invocation_id,
      new_message=testing_utils.UserContent(
          Part(
              function_response=FunctionResponse(
                  id=invocation_1_events[0].content.parts[0].function_call.id,
                  name="test_tool",
                  response={"result": "test tool update"},
              )
          ),
      ),
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(resumed_invocation_1_events)
  ) == [
      (
          root_agent.name,
          "llm response after resuming invocation 1",
      ),
      (root_agent.name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resumable_parallel_agent_escalation_short_circuits_persisted_run():
  """Runner persists fast+escalating events and marks the parent run complete."""
  fast_agent = _ParallelEscalationTestingAgent(
      name="fast_agent",
      delay=0.05,
      response_text="fast response",
  )
  escalating_agent = _ParallelEscalationTestingAgent(
      name="escalating_agent",
      delay=0.1,
      response_text="escalating response",
      escalate=True,
      emit_follow_up_after_first_event=True,
  )
  slow_agent = _ParallelEscalationTestingAgent(
      name="slow_agent",
      delay=0.5,
      response_text="slow response",
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=ParallelAgent(
              name="root_agent",
              sub_agents=[fast_agent, escalating_agent, slow_agent],
          ),
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  invocation_events = await runner.run_async("test user query")
  simplified_events = testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_events)
  )

  assert simplified_events == [
      ("root_agent", {}),
      ("fast_agent", "fast response"),
      ("escalating_agent", "escalating response"),
      ("root_agent", testing_utils.END_OF_AGENT),
  ]

  session = await runner.runner.session_service.get_session(
      app_name=runner.app_name,
      user_id="test_user",
      session_id=runner.session_id,
  )
  persisted_events = [
      event
      for event in session.events
      if event.invocation_id == invocation_events[0].invocation_id
      and event.author != "user"
  ]
  assert (
      testing_utils.simplify_resumable_app_events(
          copy.deepcopy(persisted_events)
      )
      == simplified_events
  )
  assert all(event.author != "slow_agent" for event in persisted_events)

  # A completed resumable invocation should not restart cancelled siblings.
  assert not await runner.run_async(
      invocation_id=invocation_events[0].invocation_id
  )
