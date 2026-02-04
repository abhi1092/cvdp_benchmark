# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Vertex AI model provider supporting multiple model types.

This module provides access to various models through Google Cloud's Vertex AI:
- Google Gemini models (gemini-pro, gemini-1.5-pro, gemini-2.0-flash, etc.)
- Anthropic Claude models (claude-sonnet-4, claude-3-5-sonnet, etc.)
- Other third-party models available on Vertex AI

Installation:
    pip install google-cloud-aiplatform

Authentication:
    - Set up Application Default Credentials (ADC) via gcloud:
      gcloud auth application-default login
    - Or use a service account with GOOGLE_APPLICATION_CREDENTIALS env var

Configuration:
    - VERTEX_AI_PROJECT: Your GCP project ID
    - VERTEX_AI_LOCATION: GCP region (default: us-central1 for Gemini, us-east5 for Claude)
"""

import os
import json
import logging
from typing import Optional
from src.config_manager import config
from src.model_helpers import ModelHelpers

logging.basicConfig(level=logging.INFO)

# Model name mapping for common shorthand names
# Format: model-name@date (e.g., claude-sonnet-4@20250514)
# See: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude
MODEL_ALIASES = {
    # Claude 4.x models (Anthropic) - Currently available on Vertex AI
    "claude-sonnet-4": "claude-sonnet-4@20250514",
    "claude-opus-4": "claude-opus-4@20250514",
    "claude-sonnet-4.5": "claude-sonnet-4-5@20250929",
    "claude-opus-4.1": "claude-opus-4-1@20250805",
    "claude-opus-4.5": "claude-opus-4-5@20251101",
    "claude-haiku-4.5": "claude-haiku-4-5@20251001",
    # Claude 3.x models (may be deprecated - check availability)
    "claude-3-7-sonnet": "claude-3-7-sonnet@20250219",
    "claude-3-5-sonnet": "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-haiku": "claude-3-5-haiku@20241022",
    # Gemini models (Google) - common aliases
    "gemini-pro": "gemini-1.0-pro",
    "gemini-flash": "gemini-2.0-flash-001",
    "gemini-2-flash": "gemini-2.0-flash-001",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    "gemini-1.5-pro": "gemini-1.5-pro-002",
}

# Model type detection - determines which API to use
CLAUDE_MODEL_PREFIXES = ["claude-"]
GEMINI_MODEL_PREFIXES = ["gemini-"]

# Default locations for different model types
DEFAULT_LOCATIONS = {
    "claude": "us-east5",
    "gemini": "us-central1",
    "default": "us-central1",
}


def detect_model_type(model_name: str) -> str:
    """
    Detect the model type based on the model name.

    Args:
        model_name: The model name

    Returns:
        Model type: "claude", "gemini", or "unknown"
    """
    model_lower = model_name.lower()

    for prefix in CLAUDE_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return "claude"

    for prefix in GEMINI_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return "gemini"

    return "unknown"


class VertexAI_Instance:
    """
    Vertex AI model provider supporting multiple model types.

    Automatically detects and uses the appropriate API for:
    - Google Gemini models (via GenerativeModel)
    - Anthropic Claude models (via raw_predict)
    - Other models (via raw_predict with generic format)
    """

    def __init__(
        self,
        context: str = "You are a helpful assistant.",
        key: str = None,
        model: str = None,
        base_url: str = None,
    ):
        """
        Initialize the Vertex AI model instance.

        Args:
            context: System context/prompt for the model
            key: Not used for Vertex AI (uses GCP credentials)
            model: Model name (e.g., "vertex/gemini-2.0-flash", "vertex/claude-sonnet-4")
            base_url: Not used for Vertex AI
        """
        # Suppress unused parameter warnings
        _ = key
        _ = base_url

        try:
            import vertexai
            from google.cloud import aiplatform
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform package is required for Vertex AI support. "
                "Install with: pip install google-cloud-aiplatform"
            )

        self.context = context
        self.debug = False

        # Get Vertex AI configuration
        self.project_id = config.get("VERTEX_AI_PROJECT")

        if self.project_id is None:
            raise ValueError(
                "VERTEX_AI_PROJECT must be specified in config or environment. "
                "Set it to your GCP project ID."
            )

        # Process model name
        if model is None:
            model = config.get("VERTEX_AI_MODEL", "gemini-2.0-flash-001")

        # Strip 'vertex/' prefix if present (added by model factory routing)
        if model.startswith("vertex/"):
            model = model[7:]  # Remove 'vertex/' prefix

        # Map shorthand names to full model names
        if model in MODEL_ALIASES:
            model = MODEL_ALIASES[model]

        self.model = model

        # Detect model type
        self.model_type = detect_model_type(self.model)

        # Get location - use model-specific default or config override
        config_location = config.get("VERTEX_AI_LOCATION")
        if config_location:
            self.location = config_location
        else:
            self.location = DEFAULT_LOCATIONS.get(self.model_type, DEFAULT_LOCATIONS["default"])

        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        aiplatform.init(project=self.project_id, location=self.location)

        # Store references for later use
        self._vertexai = vertexai
        self._aiplatform = aiplatform

        # Initialize model-specific client
        if self.model_type == "gemini":
            self._init_gemini_client()
        else:
            self._init_generic_client()

        logging.info("Created Vertex AI Model instance")
        logging.info(f"  - Model: {self.model}")
        logging.info(f"  - Model Type: {self.model_type}")
        logging.info(f"  - Project: {self.project_id}")
        logging.info(f"  - Location: {self.location}")

        self.set_debug(False)

    def _init_gemini_client(self):
        """Initialize the Gemini GenerativeModel client."""
        from vertexai.generative_models import GenerativeModel

        self._gemini_model = GenerativeModel(self.model)
        self._endpoint = None

    def _init_generic_client(self):
        """Initialize the generic prediction client for Claude and other models."""
        # Store the endpoint for raw predictions
        if self.model_type == "claude":
            # Claude models use the anthropic publisher
            self._endpoint = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"publishers/anthropic/models/{self.model}"
            )
        else:
            # Other models - try to infer publisher or use generic endpoint
            self._endpoint = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"publishers/google/models/{self.model}"
            )
        self._gemini_model = None

    def key(self, key: str):
        """
        Update the API key (not applicable for Vertex AI).

        Args:
            key: API key (ignored for Vertex AI)
        """
        _ = key
        logging.warning("Vertex AI uses GCP credentials, not API keys. Ignoring key parameter.")

    @property
    def requires_evaluation(self) -> bool:
        """
        Whether this model requires harness evaluation.

        Returns:
            bool: True (Vertex AI models require evaluation)
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

    def _call_gemini_api(self, system_prompt: str, user_prompt: str):
        """
        Call Gemini model via the GenerativeModel API.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt

        Returns:
            The model's response text
        """
        from vertexai.generative_models import GenerativeModel

        if self.debug:
            logging.debug(f"Calling Gemini model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {user_prompt}")

        # Create a model with system instruction
        from vertexai.generative_models import GenerationConfig

        model = GenerativeModel(
            self.model,
            system_instruction=system_prompt,
        )

        # Generate content with configurable temperature
        temperature = config.get("MODEL_TEMPERATURE", 0)
        generation_config = GenerationConfig(temperature=temperature)
        response = model.generate_content(user_prompt, generation_config=generation_config)

        # Extract text from response
        return response.text

    def _call_claude_api(self, system_prompt: str, user_prompt: str, max_tokens: int = 8192):
        """
        Call Claude model via Vertex AI raw_predict.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            max_tokens: Maximum tokens in response

        Returns:
            The model's response text
        """
        from google.cloud.aiplatform_v1 import PredictionServiceClient
        from google.api_core.client_options import ClientOptions
        from google.api import httpbody_pb2

        # Create the prediction service client
        client_options = ClientOptions(api_endpoint=f"{self.location}-aiplatform.googleapis.com")
        client = PredictionServiceClient(client_options=client_options)

        # Build the request payload (Anthropic Messages API format)
        temperature = config.get("MODEL_TEMPERATURE", 0)
        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": user_prompt}],
            "system": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.debug:
            logging.debug(f"Calling Vertex AI endpoint: {self._endpoint}")
            logging.debug(f"Payload: {payload}")

        # Make the raw prediction request
        http_body = httpbody_pb2.HttpBody(
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )

        response = client.raw_predict(
            endpoint=self._endpoint,
            http_body=http_body,
        )

        # Parse the response
        result = json.loads(response.data.decode("utf-8"))

        # Extract text from response
        content = ""
        if "content" in result:
            for block in result["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")

        return content

    def prompt(
        self,
        prompt: str,
        schema: str = None,
        prompt_log: str = "",
        files: Optional[list] = None,
        timeout: int = 60,
        category: Optional[int] = None,
    ):
        """
        Send a prompt to the Vertex AI model and get a response.

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
        _ = timeout  # Timeout handled by underlying client

        # Import and use ModelHelpers
        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)

        # Determine if we're expecting a single file (direct text mode)
        expected_single_file = files and len(files) == 1 and schema is None
        expected_file_name = files[0] if expected_single_file else None

        if self.debug:
            logging.debug(f"Requesting prompt using the model: {self.model}")
            logging.debug(f"Model type: {self.model_type}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            if files:
                logging.debug(f"Expected files: {files}")
                if expected_single_file:
                    logging.debug(f"Using direct text mode for single file: {expected_file_name}")
            logging.debug(
                f"Request parameters: model={self.model}, project={self.project_id}, location={self.location}"
            )

        # Create directories for prompt log if needed
        if prompt_log:
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(
                        system_prompt + "\n\n----------------------------------------\n" + prompt
                    )
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {str(e)}")
                raise

        try:
            # Call the appropriate API based on model type
            if self.model_type == "gemini":
                content = self._call_gemini_api(system_prompt, prompt)
            elif self.model_type == "claude":
                content = self._call_claude_api(system_prompt, prompt)
            else:
                # For unknown models, try Claude API format first (most common for third-party)
                content = self._call_claude_api(system_prompt, prompt)

            content = content.strip()

            # Print response details if debug is enabled
            if self.debug:
                logging.debug(f"  - Message: {content}")

            # Process the response using the default helper functions
            if expected_single_file:
                # For direct text response (no schema), no JSON parsing needed
                pass
            elif schema is not None and content.startswith("{") and content.endswith("}"):
                # Fix common JSON formatting issues
                content = helper.fix_json_formatting(content)

            # Call parse_model_response with the correct parameter order
            return helper.parse_model_response(content, files, expected_single_file)

        except Exception as e:
            raise ValueError(f"Unable to get response from Vertex AI model ({self.model_type}): {str(e)}")
