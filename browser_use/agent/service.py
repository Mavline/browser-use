from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import platform
import re
import textwrap
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast, TypeAlias, TypeGuard

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
)
from lmnr.sdk.decorators import observe
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.prompts import AgentMessagePrompt, PlannerPrompt, SystemPrompt
from browser_use.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentStepInfo,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import (
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
	AgentEndTelemetryEvent,
	AgentRunTelemetryEvent,
	AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
JsonPrimitive = Union[str, int, float, bool, None]
JsonDict = Dict[str, Any]
JsonList = List[Union[JsonPrimitive, JsonDict, List[Any]]]
JsonType = Union[JsonPrimitive, JsonDict, JsonList]
JsonInput: TypeAlias = Union[str, bytes, bytearray, JsonList]
JsonValue = str  # Simplify to just string
StrOrBytes = Union[str, bytes]
StrOrList = Union[str, List[Union[str, Dict[str, Any]]]]

# Определяем тип-алиас для шрифтов
FontType = Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]

def is_json_compatible(value: Any) -> TypeGuard[Union[str, bytes, bytearray]]:
	"""Check if value is JSON compatible."""
	return isinstance(value, (str, bytes, bytearray))

def ensure_str(value: Any) -> str:
	"""Safely convert any value to string."""
	if isinstance(value, (str, bytes, bytearray)):
		return str(value)
	if isinstance(value, (list, dict)):
		return json.dumps(value)
	return str(value)

def ensure_json_str(value: Any) -> Union[str, bytes, bytearray]:
	"""Ensure value is JSON compatible."""
	result = ensure_str(value)
	return cast(Union[str, bytes, bytearray], result)

def safe_json_loads(text: str) -> Any:
	"""Safely parse JSON string."""
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		return text

class TextProcessor:
	"""Helper class for text processing methods."""
	
	def __init__(self):
		self.THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
	
	def remove_think_tags(self, text: Any) -> str:
		"""Remove think tags from text."""
		try:
			text_str = ensure_str(text)
			return re.sub(self.THINK_TAGS, '', text_str)
		except Exception:
			return ""

	def process_text(self, text: Optional[Any]) -> Optional[str]:
		"""Process text by removing think tags."""
		if text is None:
			return None
		try:
			text_str = ensure_str(text)
			return self.remove_think_tags(text_str)
		except Exception:
			return None

	def extract_json(self, content: Any) -> str:
		"""Extract JSON from model output."""
		content_str = ensure_str(content)
		try:
			# Try to parse as pure JSON first
			safe_json_loads(content_str)
			return content_str
		except Exception:
			# If not pure JSON, try to extract JSON part
			json_match = re.search(r'\{.*\}', content_str, re.DOTALL)
			if json_match:
				json_str = json_match.group(0)
				# Verify it's valid JSON
				safe_json_loads(json_str)
				return json_str
			raise ValueError(f"Could not extract valid JSON from content: {content_str}")

	def parse_json(self, text: Any) -> Any:
		"""Parse JSON string into Python object."""
		try:
			text_str = ensure_str(text)
			return safe_json_loads(text_str)
		except Exception:
			return text

	def convert_to_json_value(self, text: Any) -> str:
		"""Convert any input to a JSON-compatible string value."""
		return ensure_str(text)

class Agent:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		browser: Browser | None = None,
		browser_context: BrowserContext | None = None,
		controller: Controller = Controller(),
		use_vision: bool = True,
		use_vision_for_planner: bool = False,
		save_conversation_path: Optional[str] = None,
		save_conversation_path_encoding: Optional[str] = 'utf-8',
		max_failures: int = 3,
		retry_delay: int = 10,
		system_prompt_class: Type[SystemPrompt] = SystemPrompt,
		max_input_tokens: int = 128000,
		validate_output: bool = False,
		message_context: Optional[str] = None,
		generate_gif: bool | str = True,
		sensitive_data: Optional[Dict[str, str]] = None,
		available_file_paths: Optional[list[str]] = None,
		include_attributes: list[str] = [
			'title',
			'type',
			'name',
			'role',
			'tabindex',
			'aria-label',
			'placeholder',
			'value',
			'alt',
			'aria-expanded',
		],
		max_error_length: int = 400,
		max_actions_per_step: int = 10,
		tool_call_in_content: bool = True,
		initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
		# Cloud Callbacks
		register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], None] | None = None,
		register_done_callback: Callable[['AgentHistoryList'], None] | None = None,
		tool_calling_method: Optional[str] = 'auto',
		page_extraction_llm: Optional[BaseChatModel] = None,
		planner_llm: Optional[BaseChatModel] = None,
		planner_interval: int = 1,  # Run planner every N steps
	):
		self.agent_id = str(uuid.uuid4())  # unique identifier for the agent
		self.sensitive_data = sensitive_data
		if not page_extraction_llm:
			self.page_extraction_llm = llm
		else:
			self.page_extraction_llm = page_extraction_llm
		self.available_file_paths = available_file_paths
		self.task = task
		self.use_vision = use_vision
		self.use_vision_for_planner = use_vision_for_planner
		self.llm = llm
		self.save_conversation_path = save_conversation_path
		if self.save_conversation_path and '/' not in self.save_conversation_path:
			self.save_conversation_path = f'{self.save_conversation_path}/'
		self.save_conversation_path_encoding = save_conversation_path_encoding
		self._last_result = None
		self.include_attributes = include_attributes
		self.max_error_length = max_error_length
		self.generate_gif = generate_gif

		# Initialize planner
		self.planner_llm = planner_llm
		self.planning_interval = planner_interval
		self.last_plan = None
		# Controller setup
		self.controller = controller
		self.max_actions_per_step = max_actions_per_step

		# Browser setup
		self.injected_browser = browser is not None
		self.injected_browser_context = browser_context is not None
		self.message_context = message_context

		# Initialize browser first if needed
		self.browser = browser if browser is not None else (None if browser_context else Browser())

		# Initialize browser context
		if browser_context:
			self.browser_context = browser_context
		elif self.browser:
			self.browser_context = BrowserContext(browser=self.browser, config=self.browser.config.new_context_config)
		else:
			# If neither is provided, create both new
			self.browser = Browser()
			self.browser_context = BrowserContext(browser=self.browser)

		self.system_prompt_class = system_prompt_class

		# Telemetry setup
		self.telemetry = ProductTelemetry()

		# Action and output models setup
		self._setup_action_models()
		self._set_version_and_source()
		self.max_input_tokens = max_input_tokens

		self._set_model_names()

		self.tool_calling_method = self.set_tool_calling_method(tool_calling_method)

		self.message_manager = MessageManager(
			llm=self.llm,
			task=self.task,
			action_descriptions=self.controller.registry.get_prompt_description(),
			system_prompt_class=self.system_prompt_class,
			max_input_tokens=self.max_input_tokens,
			include_attributes=self.include_attributes,
			max_error_length=self.max_error_length,
			max_actions_per_step=self.max_actions_per_step,
			message_context=self.message_context,
			sensitive_data=self.sensitive_data,
		)
		if self.available_file_paths:
			self.message_manager.add_file_paths(self.available_file_paths)
		# Step callback
		self.register_new_step_callback = register_new_step_callback
		self.register_done_callback = register_done_callback

		# Tracking variables
		self.history: AgentHistoryList = AgentHistoryList(history=[])
		self.n_steps = 1
		self.consecutive_failures = 0
		self.max_failures = max_failures
		self.retry_delay = retry_delay
		self.validate_output = validate_output
		self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None
		if save_conversation_path:
			logger.info(f'Saving conversation to {save_conversation_path}')

		self._paused = False
		self._stopped = False

		self.action_descriptions = self.controller.registry.get_prompt_description()

		self.text_processor = TextProcessor()

		self._set_version_and_source()

	def _set_version_and_source(self) -> None:
		try:
			import pkg_resources

			version = pkg_resources.get_distribution('browser-use').version
			source = 'pip'
		except Exception:
			try:
				import subprocess

				version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
				source = 'git'
			except Exception:
				version = 'unknown'
				source = 'unknown'
		logger.debug(f'Version: {version}, Source: {source}')
		self.version = version
		self.source = source

	def _set_model_names(self) -> None:
		"""Set model names safely using Any type to avoid type checking issues."""
		from typing import Any
		
		self.chat_model_library = self.llm.__class__.__name__
		self.model_name = "Unknown"
		
		llm: Any = self.llm
		planner_llm: Any = self.planner_llm
		
		try:
			if hasattr(llm, "model_name") and llm.model_name is not None:
				self.model_name = str(llm.model_name)
			elif hasattr(llm, "model") and llm.model is not None:
				self.model_name = str(llm.model)
		except Exception:
			self.model_name = "Unknown"

		self.planner_model_name = "Unknown"
		if planner_llm:
			try:
				if hasattr(planner_llm, "model_name") and planner_llm.model_name is not None:
					self.planner_model_name = str(planner_llm.model_name)
				elif hasattr(planner_llm, "model") and planner_llm.model is not None:
					self.planner_model_name = str(planner_llm.model)
			except Exception:
				self.planner_model_name = "Unknown"
		else:
			self.planner_model_name = None

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from controller's registry"""
		self.ActionModel = self.controller.registry.create_action_model()
		# Create output model with the dynamic actions
		self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

	def set_tool_calling_method(self, tool_calling_method: Optional[str]) -> Optional[str]:
		if tool_calling_method == 'auto':
			if self.chat_model_library == 'ChatGoogleGenerativeAI':
				return None
			elif self.chat_model_library == 'ChatOpenAI':
				return 'function_calling'
			elif self.chat_model_library == 'AzureChatOpenAI':
				return 'function_calling'
			else:
				return None
		else:
			return tool_calling_method

	def add_new_task(self, new_task: str) -> None:
		self.message_manager.add_new_task(new_task)

	def _check_if_stopped_or_paused(self) -> bool:
		if self._stopped or self._paused:
			logger.debug('Agent paused after getting state')
			raise InterruptedError
		return False

	@observe(name='agent.step', ignore_output=True, ignore_input=True)
	@time_execution_async('--step')
	async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
		"""Execute one step of the task"""
		logger.info(f'📍 Step {self.n_steps}')
		state = None
		model_output = None
		result: list[ActionResult] = []

		try:
			state = await self.browser_context.get_state()

			self._check_if_stopped_or_paused()
			self.message_manager.add_state_message(state, self._last_result, step_info, self.use_vision)

			# Run planner at specified intervals if planner is configured
			if self.planner_llm and self.n_steps % self.planning_interval == 0:
				plan = await self._run_planner()
				# add plan before last state message
				self.message_manager.add_plan(plan, position=-1)

			input_messages = self.message_manager.get_messages()

			self._check_if_stopped_or_paused()

			try:
				model_output = await self.get_next_action(input_messages)

				if self.register_new_step_callback:
					self.register_new_step_callback(state, model_output, self.n_steps)

				self._save_conversation(input_messages, model_output)
				self.message_manager._remove_last_state_message()  # we dont want the whole state in the chat history

				self._check_if_stopped_or_paused()

				self.message_manager.add_model_output(model_output)
			except Exception as e:
				# model call failed, remove last state message from history
				self.message_manager._remove_last_state_message()
				raise e

			result: list[ActionResult] = await self.controller.multi_act(
				model_output.action,
				self.browser_context,
				page_extraction_llm=self.page_extraction_llm,
				sensitive_data=self.sensitive_data,
				check_break_if_paused=lambda: self._check_if_stopped_or_paused(),
				available_file_paths=self.available_file_paths,
			)
			self._last_result = result

			if len(result) > 0 and result[-1].is_done:
				logger.info(f'📄 Result: {result[-1].extracted_content}')

			self.consecutive_failures = 0

		except InterruptedError:
			logger.debug('Agent paused')
			self._last_result = [
				ActionResult(
					error='The agent was paused - now continuing actions might need to be repeated', include_in_memory=True
				)
			]
			return
		except Exception as e:
			result = await self._handle_step_error(e)
			self._last_result = result

		finally:
			actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
			self.telemetry.capture(
				AgentStepTelemetryEvent(
					agent_id=self.agent_id,
					step=self.n_steps,
					actions=actions,
					consecutive_failures=self.consecutive_failures,
					step_error=[r.error for r in result if r.error] if result else ['No result'],
				)
			)
			if not result:
				return

			if state:
				self._make_history_item(model_output, state, result)

	async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
		"""Handle all types of errors that can occur during a step"""
		include_trace = logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

		if isinstance(error, (ValidationError, ValueError)):
			logger.error(f'{prefix}{error_msg}')
			if 'Max token limit reached' in error_msg:
				# cut tokens from history
				self.message_manager.max_input_tokens = self.max_input_tokens - 500
				logger.info(f'Cutting tokens from history - new max input tokens: {self.message_manager.max_input_tokens}')
				self.message_manager.cut_messages()
			elif 'Could not parse response' in error_msg:
				# give model a hint how output should look like
				error_msg += '\n\nReturn a valid JSON object with the required fields.'

			self.consecutive_failures += 1
		elif isinstance(error, RateLimitError) or isinstance(error, ResourceExhausted):
			logger.warning(f'{prefix}{error_msg}')
			await asyncio.sleep(self.retry_delay)
			self.consecutive_failures += 1
		else:
			logger.error(f'{prefix}{error_msg}')
			self.consecutive_failures += 1

		return [ActionResult(error=error_msg, include_in_memory=True)]

	def _make_history_item(
		self,
		model_output: AgentOutput | None,
		state: BrowserState,
		result: list[ActionResult],
	) -> None:
		"""Create and store history item"""
		interacted_element = None
		len_result = len(result)

		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
		else:
			interacted_elements = [None]

		state_history = BrowserStateHistory(
			url=state.url,
			title=state.title,
			tabs=state.tabs,
			interacted_element=interacted_elements,
			screenshot=state.screenshot,
		)

		history_item = AgentHistory(model_output=model_output, result=result, state=state_history)

		self.history.history.append(history_item)

	def _process_text(self, text: Optional[Any]) -> Optional[str]:
		"""Process text by removing think tags."""
		result = self.text_processor.process_text(text)
		return cast(Optional[str], result)

	def _remove_think_tags(self, text: Any) -> str:
		"""Remove think tags from text."""
		result = self.text_processor.remove_think_tags(text)
		return cast(str, result)

	def _extract_json_from_model_output(self, content: Any) -> str:
		"""Extract JSON from model output."""
		result = self.text_processor.extract_json(content)
		return cast(str, result)

	def _parse_json(self, text: Any) -> Any:
		"""Parse JSON string into Python object."""
		return self.text_processor.parse_json(text)

	def _convert_to_json_value(self, text: Any) -> str:
		"""Convert any input to a JSON-compatible value."""
		result = self.text_processor.convert_to_json_value(text)
		return cast(str, result)

	@time_execution_async('--get_next_action')
	async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get next action from the model with proper type handling."""
		from typing import Any
		
		converted_input_messages = self._convert_input_messages(input_messages, self.model_name)
		
		try:
			if self.model_name == 'deepseek-reasoner' or self.model_name.startswith('deepseek-r1'):
				output: Any = self.llm.invoke(converted_input_messages)
				processed_content = self._process_text(output.content)
				if processed_content is not None:
					output.content = processed_content
				try:
					json_content = self._extract_json_from_model_output(processed_content)
					return self.AgentOutput.model_validate_json(json_content)
				except Exception as e:
					logger.error(f'Failed to parse model output: {e}')
					raise ValueError(f'Could not parse response: {e}')
			else:
				structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
				response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
				parsed: AgentOutput | None = response['parsed']

			if parsed is None:
				raise ValueError('Could not parse response.')

			# cut the number of actions to max_actions_per_step
			parsed.action = parsed.action[: self.max_actions_per_step]
			self._log_response(parsed)
			self.n_steps += 1

			return parsed
		except Exception as e:
			logger.error(f'Model call failed: {e}')
			raise

	def _log_response(self, response: AgentOutput) -> None:
		"""Log the model's response"""
		if 'Success' in response.current_state.evaluation_previous_goal:
			emoji = '👍'
		elif 'Failed' in response.current_state.evaluation_previous_goal:
			emoji = '⚠'
		else:
			emoji = '🤷'
		logger.debug(f'🤖 {emoji} Page summary: {response.current_state.page_summary}')
		logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
		logger.info(f'🧠 Memory: {response.current_state.memory}')
		logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
		for i, action in enumerate(response.action):
			logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')

	def _save_conversation(self, input_messages: list[BaseMessage], response: Any) -> None:
		"""Save conversation history to file if path is specified"""
		if not self.save_conversation_path:
			return

		# create folders if not exists
		os.makedirs(os.path.dirname(self.save_conversation_path), exist_ok=True)

		with open(
			self.save_conversation_path + f'_{self.n_steps}.txt',
			'w',
			encoding=self.save_conversation_path_encoding,
		) as f:
			self._write_messages_to_file(f, input_messages)
			self._write_response_to_file(f, response)

	def _write_messages_to_file(self, f: Any, messages: list[BaseMessage]) -> None:
		"""Write messages to conversation file"""
		for message in messages:
			f.write(f' {message.__class__.__name__} \n')

			if isinstance(message.content, list):
				for item in message.content:
					if isinstance(item, dict) and item.get('type') == 'text':
						f.write(item['text'].strip() + '\n')
			elif isinstance(message.content, str):
				try:
					content = json.loads(message.content)
					f.write(json.dumps(content, indent=2) + '\n')
				except json.JSONDecodeError:
					f.write(message.content.strip() + '\n')

			f.write('\n')

	def _write_response_to_file(self, f: Any, response: Any) -> None:
		"""Write model response to conversation file"""
		f.write(' RESPONSE\n')
		f.write(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

	def _log_agent_run(self) -> None:
		"""Log the agent run"""
		logger.info(f'🚀 Starting task: {self.task}')

		logger.debug(f'Version: {self.version}, Source: {self.source}')
		self.telemetry.capture(
			AgentRunTelemetryEvent(
				agent_id=self.agent_id,
				use_vision=self.use_vision,
				task=self.task,
				model_name=self.model_name,
				chat_model_library=self.chat_model_library,
				version=self.version,
				source=self.source,
			)
		)

	@observe(name='agent.run', ignore_output=True)
	async def run(self, max_steps: int = 100) -> AgentHistoryList:
		"""Execute the task with maximum number of steps"""
		try:
			self._log_agent_run()

			# Execute initial actions if provided
			if self.initial_actions:
				result = await self.controller.multi_act(
					self.initial_actions,
					self.browser_context,
					check_for_new_elements=False,
					page_extraction_llm=self.page_extraction_llm,
					check_break_if_paused=lambda: self._check_if_stopped_or_paused(),
					available_file_paths=self.available_file_paths,
				)
				self._last_result = result

			for step in range(max_steps):
				if self._too_many_failures():
					break

				# Check control flags before each step
				if not await self._handle_control_flags():
					break

				await self.step()

				if self.history.is_done():
					if self.validate_output and step < max_steps - 1:
						if not await self._validate_output():
							continue

					logger.info('✅ Task completed successfully')
					if self.register_done_callback:
						self.register_done_callback(self.history)
					break
			else:
				logger.info('❌ Failed to complete task in maximum steps')

			return self.history
		finally:
			self.telemetry.capture(
				AgentEndTelemetryEvent(
					agent_id=self.agent_id,
					success=self.history.is_done(),
					steps=self.n_steps,
					max_steps_reached=self.n_steps >= max_steps,
					errors=self.history.errors(),
				)
			)

			if not self.injected_browser_context:
				await self.browser_context.close()

			if not self.injected_browser and self.browser:
				await self.browser.close()

			if self.generate_gif:
				output_path: str = 'agent_history.gif'
				if isinstance(self.generate_gif, str):
					output_path = self.generate_gif

				self.create_history_gif(output_path=output_path)

	def _too_many_failures(self) -> bool:
		"""Check if we should stop due to too many failures"""
		if self.consecutive_failures >= self.max_failures:
			logger.error(f'❌ Stopping due to {self.max_failures} consecutive failures')
			return True
		return False

	async def _handle_control_flags(self) -> bool:
		"""Handle pause and stop flags. Returns True if execution should continue."""
		if self._stopped:
			logger.info('Agent stopped')
			return False

		while self._paused:
			await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
			if self._stopped:  # Allow stopping while paused
				return False
		return True

	async def _validate_output(self) -> bool:
		"""Validate the output of the last action is what the user wanted"""
		system_msg = (
			f'You are a validator of an agent who interacts with a browser. '
			f'Validate if the output of last action is what the user wanted and if the task is completed. '
			f'If the task is unclear defined, you can let it pass. But if something is missing or the image does not show what was requested dont let it pass. '
			f'Try to understand the page and help the model with suggestions like scroll, do x, ... to get the solution right. '
			f'Task to validate: {self.task}. Return a JSON object with 2 keys: is_valid and reason. '
			f'is_valid is a boolean that indicates if the output is correct. '
			f'reason is a string that explains why it is valid or not.'
			f' example: {{"is_valid": false, "reason": "The user wanted to search for "cat photos", but the agent searched for "dog photos" instead."}}'
		)

		if self.browser_context.session:
			state = await self.browser_context.get_state()
			content = AgentMessagePrompt(
				state=state,
				result=self._last_result,
				include_attributes=self.include_attributes,
				max_error_length=self.max_error_length,
			)
			msg = [SystemMessage(content=system_msg), content.get_user_message(self.use_vision)]
		else:
			# if no browser session, we can't validate the output
			return True

		class ValidationResult(BaseModel):
			"""
			Validation results.
			"""
			is_valid: bool
			reason: str

		validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
		response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
		parsed: ValidationResult = response['parsed']
		is_valid = parsed.is_valid
		if not is_valid:
			logger.info(f'❌ Validator decision: {parsed.reason}')
			msg = f'The output is not yet correct. {parsed.reason}.'
			self._last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
		else:
			logger.info(f'✅ Validator decision: {parsed.reason}')
		return is_valid

	async def rerun_history(
		self,
		history: AgentHistoryList,
		max_retries: int = 3,
		skip_failures: bool = True,
		delay_between_actions: float = 2.0,
	) -> list[ActionResult]:
		"""
		Rerun a saved history of actions with error handling and retry logic.

		Args:
				history: The history to replay
				max_retries: Maximum number of retries per action
				skip_failures: Whether to skip failed actions or stop execution
				delay_between_actions: Delay between actions in seconds

		Returns:
				List of action results
		"""
		# Execute initial actions if provided
		if self.initial_actions:
			await self.controller.multi_act(
				self.initial_actions,
				self.browser_context,
				check_for_new_elements=False,
				page_extraction_llm=self.page_extraction_llm,
				check_break_if_paused=lambda: self._check_if_stopped_or_paused(),
				available_file_paths=self.available_file_paths,
			)

		results = []

		for i, history_item in enumerate(history.history):
			goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
			logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

			if (
				not history_item.model_output
				or not history_item.model_output.action
				or history_item.model_output.action == [None]
			):
				logger.warning(f'Step {i + 1}: No action to replay, skipping')
				results.append(ActionResult(error='No action to replay'))
				continue

			retry_count = 0
			while retry_count < max_retries:
				try:
					result = await self._execute_history_step(history_item, delay_between_actions)
					results.extend(result)
					break

				except Exception as e:
					retry_count += 1
					if retry_count == max_retries:
						error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
						logger.error(error_msg)
						if not skip_failures:
							results.append(ActionResult(error=error_msg))
							raise RuntimeError(error_msg)
					else:
						logger.warning(f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...')
						await asyncio.sleep(delay_between_actions)

		return results

	async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
		"""Execute a single step from history with element validation"""
		state = await self.browser_context.get_state()
		if not state or not history_item.model_output:
			raise ValueError('Invalid state or model output')
		updated_actions = []
		for i, action in enumerate(history_item.model_output.action):
			updated_action = await self._update_action_indices(
				history_item.state.interacted_element[i],
				action,
				state,
			)
			updated_actions.append(updated_action)

			if updated_action is None:
				raise ValueError(f'Could not find matching element {i} in current page')

		result = await self.controller.multi_act(
			updated_actions,
			self.browser_context,
			page_extraction_llm=self.page_extraction_llm,
			check_break_if_paused=lambda: self._check_if_stopped_or_paused(),
		)

		await asyncio.sleep(delay)
		return result

	async def _update_action_indices(
		self,
		historical_element: Optional[DOMHistoryElement],
		action: ActionModel,  # Type this properly based on your action model
		current_state: BrowserState,
	) -> Optional[ActionModel]:
		"""
		Update action indices based on current page state.
		Returns updated action or None if element cannot be found.
		"""
		if not historical_element or not current_state.element_tree:
			return action

		current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, current_state.element_tree)

		if not current_element or current_element.highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != current_element.highlight_index:
			action.set_index(current_element.highlight_index)
			logger.info(f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}')

		return action

	async def load_and_rerun(self, history_file: Optional[str | Path] = None, **kwargs) -> list[ActionResult]:
		"""
		Load history from file and rerun it.

		Args:
				history_file: Path to the history file
				**kwargs: Additional arguments passed to rerun_history
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: Optional[str | Path] = None) -> None:
		"""Save the history to a file"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path)

	def create_history_gif(
		self,
		output_path: str = 'agent_history.gif',
		duration: int = 3000,
		show_goals: bool = True,
		show_task: bool = True,
		show_logo: bool = False,
		font_size: int = 40,
		title_font_size: int = 56,
		goal_font_size: int = 44,
		margin: int = 40,
		line_spacing: float = 1.5,
	) -> None:
		"""Create a GIF from the agent's history with overlaid task and goal text."""
		if not self.history.history:
			logger.warning('No history to create GIF from')
			return

		images = []
		# if history is empty or first screenshot is None, we can't create a gif
		if not self.history.history or not self.history.history[0].state.screenshot:
			logger.warning('No history or first screenshot to create GIF from')
			return

		# Try to load nicer fonts
		try:
			# Try different font options in order of preference
			font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
			font_loaded = False

			for font_name in font_options:
				try:
					if platform.system() == 'Windows':
						# Need to specify the abs font path on Windows
						font_name = os.path.join(os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts'), font_name + '.ttf')
					regular_font = ImageFont.truetype(font_name, font_size)
					title_font = ImageFont.truetype(font_name, title_font_size)
					goal_font = ImageFont.truetype(font_name, goal_font_size)
					font_loaded = True
					break
				except OSError:
					continue

			if not font_loaded:
				raise OSError('No preferred fonts found')

		except OSError:
			regular_font = ImageFont.load_default()
			title_font = ImageFont.load_default()

			goal_font = regular_font

		# Load logo if requested
		logo = None
		if show_logo:
			try:
				logo = Image.open('./static/browser-use.png')
				# Resize logo to be small (e.g., 40px height)
				logo_height = 150
				aspect_ratio = logo.width / logo.height
				logo_width = int(logo_height * aspect_ratio)
				logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
			except Exception as e:
				logger.warning(f'Could not load logo: {e}')

		# Create task frame if requested
		if show_task and self.task:
			task_frame = self._create_task_frame(
				self.task,
				self.history.history[0].state.screenshot,
				title_font,
				regular_font,
				logo,
				line_spacing,
			)
			images.append(task_frame)

		# Process each history item
		for i, item in enumerate(self.history.history, 1):
			if not item.state.screenshot:
				continue

			# Convert base64 screenshot to PIL Image
			img_data = base64.b64decode(item.state.screenshot)
			image = Image.open(io.BytesIO(img_data))

			if show_goals and item.model_output:
				image = self._add_overlay_to_image(
					image=image,
					step_number=i,
					goal_text=item.model_output.current_state.next_goal,
					regular_font=regular_font,
					title_font=title_font,
					margin=margin,
					logo=logo,
				)

			images.append(image)

		if images:
			# Save the GIF
			images[0].save(
				output_path,
				save_all=True,
				append_images=images[1:],
				duration=duration,
				loop=0,
				optimize=False,
			)
			logger.info(f'Created GIF at {output_path}')
		else:
			logger.warning('No images found in history to create GIF')

	def _get_text_height(self, font: FontType, text: str) -> int:
		"""Get height of text using getbbox."""
		font_ft = cast(ImageFont.FreeTypeFont, font)
		bbox = font_ft.getbbox(text)
		return int(bbox[3] - bbox[1])

	def _create_task_frame(
		self,
		task: str,
		first_screenshot: str,
		title_font: FontType,
		regular_font: FontType,
		logo: Optional[Image.Image] = None,
		line_spacing: float = 1.5,
	) -> Image.Image:
		"""Create initial frame with task description."""
		# Decode base64 screenshot
		screenshot_bytes = base64.b64decode(first_screenshot)
		screenshot = Image.open(BytesIO(screenshot_bytes))
		
		# Get dimensions
		width = screenshot.width
		height = screenshot.height + 400  # Extra space for text
		
		# Create new image with white background
		image = Image.new('RGB', (width, height), 'white')
		image.paste(screenshot, (0, 400))
		
		draw = ImageDraw.Draw(image)
		
		# Add task text
		task_lines = textwrap.wrap(task, width=60)
		y_text = 50
		
		# Add title
		title_font_ft = cast(ImageFont.FreeTypeFont, title_font)
		regular_font_ft = cast(ImageFont.FreeTypeFont, regular_font)
		
		draw.text((50, y_text), "Task:", font=title_font_ft, fill='black')
		y_text += self._get_text_height(title_font_ft, "Task:") * line_spacing
		
		# Add task description
		for line in task_lines:
			draw.text((50, y_text), line, font=regular_font_ft, fill='black')
			y_text += self._get_text_height(regular_font_ft, line) * line_spacing
		
		# Add logo if provided
		if logo:
			# Calculate position (bottom right)
			logo_x = width - logo.width - 50
			logo_y = 400 - logo.height - 50
			image.paste(logo, (logo_x, logo_y), logo)
		
		return image

	def _add_overlay_to_image(
		self,
		image: Image.Image,
		step_number: int,
		goal_text: str,
		regular_font: FontType,
		title_font: FontType,
		margin: int,
		logo: Optional[Image.Image] = None,
		display_step: bool = True,
		text_color: tuple[int, int, int, int] = (255, 255, 255, 255),
		text_box_color: tuple[int, int, int, int] = (0, 0, 0, 255),
	) -> Image.Image:
		"""Add text overlay to image."""
		# Create copy of image
		result = image.copy()
		draw = ImageDraw.Draw(result)
		
		# Calculate text dimensions
		title_font_ft = cast(ImageFont.FreeTypeFont, title_font)
		regular_font_ft = cast(ImageFont.FreeTypeFont, regular_font)
		
		# Get text dimensions
		if display_step:
			step_text = f"Step {step_number}"
			step_width = self._get_text_width(draw, step_text, title_font_ft)
			step_height = self._get_text_height(title_font_ft, step_text)
		
		# Wrap goal text
		max_text_width = image.width - (2 * margin)
		wrapped_text = self._wrap_text(goal_text, regular_font_ft, max_text_width, draw)
		text_lines = wrapped_text.split('\n')
		
		# Calculate total text height
		total_height = 0
		if display_step:
			total_height += step_height + margin
		
		line_height = self._get_text_height(regular_font_ft, text_lines[0])
		total_height += (len(text_lines) * line_height) + (2 * margin)
		
		# Draw semi-transparent black background
		draw.rectangle(
			[(0, 0), (image.width, total_height)],
			fill=text_box_color,
		)
		
		# Draw text
		y = margin
		if display_step:
			draw.text((margin, y), step_text, font=title_font_ft, fill=text_color)
			y += step_height + margin
		
		for line in text_lines:
			draw.text((margin, y), line, font=regular_font_ft, fill=text_color)
			y += line_height
		
		# Add logo if provided
		if logo:
			# Calculate position (top right)
			logo_x = image.width - logo.width - margin
			logo_y = margin
			result.paste(logo, (logo_x, logo_y), logo)
		
		return result

	def _get_text_dimensions(
		self, 
		draw: ImageDraw.ImageDraw, 
		text: str, 
		font: FontType
	) -> tuple[int, int]:
		"""Get width and height of text."""
		font_ft = cast(ImageFont.FreeTypeFont, font)
		bbox = font_ft.getbbox(text)
		return (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))

	def _get_text_width(
		self, 
		draw: ImageDraw.ImageDraw, 
		text: str, 
		font: FontType
	) -> int:
		"""Get width of text."""
		return self._get_text_dimensions(draw, text, font)[0]

	def _wrap_text(
		self, 
		text: str, 
		font: FontType, 
		max_width: int, 
		draw: ImageDraw.ImageDraw
	) -> str:
		"""Wrap text to fit within max_width."""
		font_ft = cast(ImageFont.FreeTypeFont, font)
		words = text.split()
		lines = []
		current_line = []
		
		for word in words:
			# Try adding new word
			test_line = ' '.join(current_line + [word])
			width = self._get_text_width(draw, test_line, font_ft)
			
			if width <= max_width:
				current_line.append(word)
			else:
				# Line is full, start new line
				if current_line:
					lines.append(' '.join(current_line))
					current_line = [word]
				else:
					# Single word is too long, force it on its own line
					lines.append(word)
					current_line = []
		
		# Add remaining words
		if current_line:
			lines.append(' '.join(current_line))
		
		return '\n'.join(lines)

	def pause(self) -> None:
		"""Pause the agent before the next step"""
		logger.info('�� pausing Agent ')
		self._paused = True

	def resume(self) -> None:
		"""Resume the agent"""
		logger.info('▶️ Agent resuming')
		self._paused = False

	def stop(self) -> None:
		"""Stop the agent"""
		logger.info('⏹️ Agent stopping')
		self._stopped = True

	def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
		"""Convert dictionary-based actions to ActionModel instances"""
		converted_actions = []
		action_model = self.ActionModel
		for action_dict in actions:
			# Each action_dict should have a single key-value pair
			action_name = next(iter(action_dict))
			params = action_dict[action_name]

			# Get the parameter model for this action from registry
			action_info = self.controller.registry.registry.actions[action_name]
			param_model = action_info.param_model

			# Create validated parameters using the appropriate param model
			validated_params = param_model(**params)

			# Create ActionModel instance with the validated parameters
			action_model = self.ActionModel(**{action_name: validated_params})
			converted_actions.append(action_model)

		return converted_actions

	async def _run_planner(self) -> Optional[str]:
		"""Run the planner to analyze state and suggest next steps"""
		if not self.planner_llm:
			return None

		planner_messages = [
			PlannerPrompt(self.action_descriptions).get_system_message(),
			*self.message_manager.get_messages()[1:],
		]

		if not self.use_vision_for_planner and self.use_vision:
			last_state_message = planner_messages[-1]
			new_msg = ''
			if isinstance(last_state_message.content, list):
				for msg in last_state_message.content:
					if msg['type'] == 'text':
						new_msg += msg['text']
					elif msg['type'] == 'image_url':
						continue
			else:
				new_msg = last_state_message.content

			planner_messages[-1] = HumanMessage(content=new_msg)

		planner_messages = self._convert_input_messages(planner_messages, self.planner_model_name)
		response = await self.planner_llm.ainvoke(planner_messages)
		plan = response.content

		if self.planner_model_name == 'deepseek-reasoner':
			plan = self._remove_think_tags(plan)

		try:
			plan_json = self._parse_json(plan)
			logger.info(f'Planning Analysis:\n{json.dumps(plan_json, indent=4)}')
		except Exception as e:
			logger.info(f'Planning Analysis:\n{plan}')

		return cast(Optional[str], plan)

	def _convert_input_messages(self, input_messages: list[BaseMessage], model_name: Optional[str]) -> list[BaseMessage]:
		"""Convert input messages to a format that is compatible with the model."""
		if model_name is None:
			return input_messages
		if model_name == 'deepseek-reasoner' or model_name.startswith('deepseek-r1'):
			converted_input_messages = self.message_manager.convert_messages_for_non_function_calling_models(input_messages)
			merged_input_messages = self.message_manager.merge_successive_messages(converted_input_messages, HumanMessage)
			merged_input_messages = self.message_manager.merge_successive_messages(merged_input_messages, AIMessage)
			return merged_input_messages
		return input_messages
