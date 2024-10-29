from openai import OpenAI
import logging
from typing import List, Dict, Union, Optional

class OpenAIModel:
    """
    Class to interact with the OpenAI API for chat-based model conversations.
    
    Attributes:
        client (OpenAI): The OpenAI API client initialized with the API key.
        model_name (str): The name of the model to use.
        max_tokens (int): The maximum number of tokens determined based on the model.
        _messages (list): List to store the conversation message history.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", console_debug: bool = False):
        """
        Initializes an instance of OpenAIModel with the specified model and API client.
        
        Parameters:
            api_key (str): The API key to access the OpenAI API.
            model_name (str): The model name to use. Default is 'gpt-4o-mini'.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        email_handler = logging.FileHandler("logs/openai_model.log")
        email_handler.setLevel(logging.DEBUG)
        email_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        email_handler.setFormatter(email_formatter)
        self.logger.addHandler(email_handler)

        if console_debug:
            email_handler_console = logging.StreamHandler()
            email_handler_console.setLevel(logging.DEBUG)
            email_handler_console.setFormatter(email_formatter)
            self.logger.addHandler(email_handler_console)

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = self.get_model_max_tokens(model_name)
        self.logger.info(f"Initialized OpenAIModel with model '{model_name}', max tokens: {self.max_tokens}")
        self._messages: List[Dict[str, str]] = []
        self._response: Optional[Dict] = None

    def get_model_max_tokens(self, model_name: str) -> int:
        """
        Retrieves the maximum number of tokens for the specified model.
        
        Parameters:
            model_name (str): The name of the model to get the token count for.
        
        Returns:
            int: The maximum number of tokens for the model.
        """
        # Dictionary of token limits per model
        token_limits = {
            "gpt-4o-mini": 16384,
            "gpt-3.5-turbo": 4096
        }
        max_tokens = token_limits.get(model_name)
        if not max_tokens:
            self.logger.warning(f"Model '{model_name}' not recognized. Using default max tokens (4096).")
            return 4096
        self.logger.debug(f"Max tokens for model '{model_name}': {max_tokens}")
        return max_tokens

    def list_available_models(self) -> Optional[List[str]]:
        """
        Retrieves a list of available models via the OpenAI API.

        Returns:
            Optional[List[str]]: A list of available model IDs or None in case of an error.
        """
        try:
            self.logger.debug("Attempting to retrieve available models from OpenAI API.")
            models = self.client.models.list()
            available_models = [model['id'] for model in models['data']]
            self.logger.info(f"Available models retrieved successfully: {available_models}")
            return available_models
        except Exception as e:
            self.logger.error(f"Error retrieving models: {e}")
            return None

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the conversation history.

        Parameters:
            role (str): The role of the message sender (e.g., 'user' or 'assistant').
            content (str): The content of the message.
        """
        self.logger.debug(f"Adding message to history - Role: {role}, Content: {content[:50]}...")  # Truncated log
        if role not in ("user", "assistant", "system"):
            self.logger.error(f"Invalid role '{role}' provided.")
            raise ValueError("Invalid role. Choose from 'user', 'assistant', or 'system'.")
        if not content.strip():
            self.logger.error("Attempted to add empty content message.")
            raise ValueError("Content must be a non-empty string.")
        
        # Add the validated message to the history
        self._messages.append({"role": role, "content": content.strip()})
        self.logger.debug(f"Message added. Total messages: {len(self._messages)}")

    def chat(self, temperature: float = 0.7, top_p: float = 1.0, n: int = 1, stop: Optional[Union[str, List[str]]] = None) -> Optional[str]:
        """
        Sends the conversation history to the OpenAI API and obtains a response with custom parameters.

        Parameters:
            temperature (float): Controls randomness. Lower values make the result more deterministic.
            top_p (float): Controls diversity via nucleus sampling. Values <1.0 consider fewer options.
            n (int): Number of responses to generate.
            stop (Optional[Union[str, List[str]]]): One or more strings where the model will stop generating tokens.

        Returns:
            Optional[str]: The response message content or None in case of an error.
        """
        self.logger.info(f"Sending messages to OpenAI API - Model: {self.model_name}, Temp: {temperature}, top_p: {top_p}, n: {n}, stop: {stop}")
        self.logger.debug(f"Message payload: {self._messages}")

        try:
            # Sending chat completion request to the API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop
            )
            response_content = response.choices[0].message.content.strip()
            self.logger.debug(f"Response content: {response_content[:50]}...")  # Truncated log
            # Add response to history
            self.add_message("assistant", response_content)
            self._response = response
            self.logger.info("Response successfully received and added to history.")
            return response_content
        except Exception as e:
            self.logger.error(f"Error during chat completion: {e}")
            return None

    def reset_conversation(self) -> None:
        """
        Resets the conversation history and response attributes.
        """
        self.logger.info("Resetting conversation history and response.")
        self._messages.clear()
        self._response = None
        self.logger.debug("Conversation and response have been reset.")

    def get_last_response(self) -> Optional[str]:
        """
        Retrieves the last response from the OpenAI API if available.

        Returns:
            Optional[str]: The content of the last response or None if unavailable.
        """
        self.logger.debug("Attempting to retrieve the last response.")
        if self._response:
            response_content = self._response['choices'][0]['message']['content']
            self.logger.info("Last response retrieved successfully.")
            return response_content
        self.logger.warning("No response available to return.")
        return None
