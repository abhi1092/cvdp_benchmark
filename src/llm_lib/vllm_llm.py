# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import openai
import os
import logging
import json
from typing import Optional, Any, Dict
from src.config_manager import config
from src.model_helpers import ModelHelpers

logging.basicConfig(level=logging.INFO)

RETRY_CODES = [429, 502, 503, 504]
WAIT_TIME = 1.5  # Seconds to wait between API calls for error resilience

class VLLM_Instance:
    """
    vLLM model provider that uses the OpenAI-compatible API provided by vLLM.

    vLLM (https://github.com/vllm-project/vllm) provides a high-throughput and memory-efficient
    inference engine for LLMs. It exposes an OpenAI-compatible API server that can be used
    with the OpenAI Python client.

    Configuration:
        - VLLM_BASE_URL: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        - VLLM_API_KEY: Optional API key for the vLLM server (defaults to "EMPTY")
        - VLLM_MODEL: Model name to use (must match the model loaded in vLLM server)

    Example:
        # Start vLLM server:
        # python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B

        # In your config or environment:
        # VLLM_BASE_URL=http://localhost:8000/v1
        # VLLM_MODEL=meta-llama/Llama-3-8B
    """

    def __init__(self, context: str = "You are a helpful assistant.", key: str = None, model: str = None, base_url: str = None):
        """
        Initialize the vLLM model instance.

        Args:
            context: System context/prompt for the model
            key: API key (optional, defaults to "EMPTY" for vLLM)
            model: Model name (must match the model loaded in vLLM server)
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        """
        if model is None:
            model = config.get("VLLM_MODEL")

        if model is None:
            raise ValueError("VLLM_MODEL must be specified in config or passed as parameter")

        # Strip 'vllm/' prefix if present (added by model factory routing)
        if model.startswith("vllm/"):
            model = model[5:]  # Remove 'vllm/' prefix

        self.context = context
        self.model = model
        self.debug = False

        # Get vLLM configuration
        vllm_base_url = base_url or config.get("VLLM_BASE_URL")
        vllm_api_key = key or config.get("VLLM_API_KEY", "EMPTY")

        if vllm_base_url is None:
            raise ValueError("VLLM_BASE_URL must be specified in config or passed as parameter")

        # Create OpenAI client configured for vLLM
        self.chat = openai.OpenAI(
            api_key=vllm_api_key,
            base_url=vllm_base_url
        )

        logging.info(f"Created vLLM Model instance")
        logging.info(f"  - Model: {self.model}")
        logging.info(f"  - Base URL: {vllm_base_url}")

        self.set_debug(False)  # Debug off by default

    def key(self, key: str):
        """
        Update the API key.

        Args:
            key: New API key to use
        """
        base_url = config.get("VLLM_BASE_URL")
        self.chat = openai.OpenAI(api_key=key, base_url=base_url)

    @property
    def requires_evaluation(self) -> bool:
        """
        Whether this model requires harness evaluation.

        Returns:
            bool: True (vLLM models require evaluation)
        """
        return True

    def set_debug(self, debug: bool = True) -> None:
        """
        Enable or disable debug mode.

        Args:
            debug: Whether to enable debug mode (default: True)
        """
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    def prompt(self, prompt: str, schema: str = None, prompt_log: str = "",
               files: Optional[list] = None, timeout: int = 60, category: Optional[int] = None):
        """
        Send a prompt to the vLLM model and get a response.

        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds for the API call (default: 60)
            category: Optional integer indicating the category/problem ID

        Returns:
            The model's response as text
        """
        if self.chat is None:
            raise ValueError("Unable to detect Chat Model")

        # Import and use ModelHelpers
        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)

        # Use timeout from config if not specified
        if timeout == 60:
            timeout = config.get("MODEL_TIMEOUT", 60)

        # Determine if we're expecting a single file (direct text mode)
        expected_single_file = files and len(files) == 1 and schema is None
        expected_file_name = files[0] if expected_single_file else None

        if self.debug:
            logging.debug(f"Requesting prompt using the model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            if files:
                logging.debug(f"Expected files: {files}")
                if expected_single_file:
                    logging.debug(f"Using direct text mode for single file: {expected_file_name}")
            logging.debug(f"Request parameters: model={self.model}, timeout={timeout}")

        # Create directories for prompt log if needed
        if prompt_log:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)

                # Write to a temporary file first
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(system_prompt + "\n\n----------------------------------------\n" + prompt)

                # Atomic rename to final file
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {str(e)}")
                # Don't continue if we can't write the log file
                raise

        try:
            # Create a chat completion request
            response = self.chat.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                timeout=timeout
            )

            # Print response details if debug is enabled
            if self.debug:
                logging.debug(f"Response received:\n{response}")

            for choice in response.choices:
                message = choice.message
                if self.debug:
                    logging.debug(f"  - Message: {message.content}")

                content = message.content.strip()

                # Process the response using the default helper functions
                if expected_single_file:
                    # For direct text response (no schema), no JSON parsing needed
                    pass
                elif schema is not None and content.startswith('{') and content.endswith('}'):
                    # Fix common JSON formatting issues
                    content = helper.fix_json_formatting(content)

                # Call parse_model_response with the correct parameter order
                return helper.parse_model_response(content, files, expected_single_file)

        except Exception as e:
            # Raise a specific error like the internal implementations
            raise ValueError(f"Unable to get response from vLLM model: {str(e)}")
