# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import openai
import os
import logging
import json
import re
import time
import requests
from typing import Optional, Any, Dict
from src.config_manager import config
from src.model_helpers import ModelHelpers

logging.basicConfig(level=logging.INFO)

RETRY_CODES = [429, 502, 503, 504]
WAIT_TIME = 1.5  # Seconds to wait between API calls for error resilience

class OpenAI_Responses_Instance:

    # ----------------------------------------
    # - Initiate the Model
    # ----------------------------------------

    def __init__(self, context : str = "You are a helpful assistant.", key = None, model = None, base_url = None):
        if model is None:
            model = config.get("DEFAULT_MODEL")

        self.context = context
        self.model   = model
        self.debug   = False

        # Check for OpenRouter configuration first
        openrouter_key = config.get("OPENROUTER_API_KEY")
        openrouter_base_url = config.get("OPENROUTER_BASE_URL")
        openrouter_model = config.get("OPENROUTER_MODEL")

        # Determine which API to use
        use_openrouter = False
        if openrouter_key and (openrouter_model or model):
            # If OpenRouter is configured and we have a model, use OpenRouter
            use_openrouter = True
            api_key = openrouter_key
            if openrouter_model and model == config.get("DEFAULT_MODEL"):
                # Use OpenRouter model if we're using the default model
                self.model = openrouter_model
            base_url = base_url or openrouter_base_url
        else:
            # Fall back to OpenAI
            api_key = config.get("OPENAI_USER_KEY")

        # Override with provided key if given
        if key is not None:
            api_key = key

        if api_key is None:
            raise ValueError("Unable to create Chat Model: No API key found")

        # Create client with appropriate configuration
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)

        provider = "OpenRouter" if use_openrouter else "OpenAI"
        logging.info(f"Created {provider} Model. Using model: {self.model}")
        if base_url:
            logging.info(f"Using custom base URL: {base_url}")

        self.set_debug(False)  # Debug off by default

    # ----------------------------------------
    # - Assign a new Key
    # ----------------------------------------

    def key(self, key):
        self.client = openai.OpenAI(api_key=key)

    @property
    def requires_evaluation(self) -> bool:
        """
        Whether this model requires harness evaluation.
        
        Default is True for backward compatibility.
        
        Returns:
            bool: True (standard models require evaluation)
        """
        return True

    # ----------------------------------------
    # - Prompt a new Request
    # ----------------------------------------
    
    def set_debug(self, debug: bool = True) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            debug: Whether to enable debug mode (default: True)
        """
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")
    
    def prompt(self, prompt, schema: str = None, prompt_log: str = "", files: Optional[list] = None, timeout: int = 60, category: Optional[int] = None):
        """
        Send a prompt to the OpenAI model and get a response.
        
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
        if not hasattr(self, 'client') or self.client is None:
            raise ValueError("Unable to detect OpenAI client")

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
            # Use the /responses API
            # Respect request timeout via per-call client options when supported
            temperature = config.get("MODEL_TEMPERATURE", 0)
            client_for_call = getattr(self.client, 'with_options', lambda **_: self.client)(timeout=timeout)
            resp = client_for_call.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=prompt,
                temperature=temperature,
            )

            if self.debug:
                logging.debug(f"Response received:\n{resp}")

            # Try to extract plain text from the responses object
            content_text = None

            # Preferred convenience property (available in recent SDKs)
            if hasattr(resp, 'output_text') and isinstance(getattr(resp, 'output_text'), str):
                content_text = resp.output_text.strip()

            # Fallbacks for different SDK object shapes
            if content_text is None:
                try:
                    # New responses objects often have .output -> list of items with .content -> list with .text.value
                    output = getattr(resp, 'output', None)
                    if output:
                        texts = []
                        for item in output:
                            parts = getattr(item, 'content', [])
                            for p in parts:
                                txt = getattr(p, 'text', None)
                                if isinstance(txt, str):
                                    texts.append(txt)
                                elif txt is not None and hasattr(txt, 'value'):
                                    texts.append(getattr(txt, 'value'))
                        if texts:
                            content_text = "\n".join(texts).strip()
                except Exception:
                    pass

            # Last resort: some SDKs still expose choices-style objects
            if content_text is None and hasattr(resp, 'choices'):
                try:
                    message = resp.choices[0].message
                    content_text = message.content.strip() if hasattr(message, 'content') else None
                except Exception:
                    pass

            if not content_text:
                raise ValueError("Empty response content from OpenAI /responses API")

            # Process the response using the default helper functions
            if expected_single_file:
                # For direct text response (no schema), no JSON parsing needed
                parsed = helper.parse_model_response(content_text, files, expected_single_file)
                return parsed
            
            if schema is not None and content_text.startswith('{') and content_text.endswith('}'):
                # Fix common JSON formatting issues
                content_text = helper.fix_json_formatting(content_text)

            return helper.parse_model_response(content_text, files, expected_single_file)

        except Exception as e:
            raise ValueError(f"Unable to get response from OpenAI model (/responses): {str(e)}")
