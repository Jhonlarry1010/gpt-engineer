"""
Gemini AI adapter for GPT Engineer.

This module provides an adapter for Google's Gemini API to be used in GPT Engineer.
It follows a similar interface to the OpenAI adapter but uses the Google Generative AI client.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

from gpt_engineer.core.base_ai import BaseAI
from gpt_engineer.core.files_dict import FilesDict

logger = logging.getLogger(__name__)

class GeminiAI(BaseAI):
    """
    AI adapter for Google's Gemini models.
    
    This class provides an interface to Gemini models similar to the OpenAI adapter.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Gemini AI.
        
        Parameters
        ----------
        model_name : str
            The name of the Gemini model to use.
        temperature : float
            The temperature for generation (controls randomness).
        api_key : Optional[str]
            The Gemini API key. If not provided, it will be read from
            the GEMINI_API_KEY environment variable.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.vision = False  # Adjust based on model capabilities
        
        # Load API key from environment if not provided
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()  # Load from .env file if exists
            self.api_key = os.getenv("GEMINI_API_KEY")
            
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided. Set the GEMINI_API_KEY environment variable or pass the key directly."
            )
            
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8000,  # Adjust based on model limits
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
            )
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise
        
        # Track token usage (note: Gemini measures differently than OpenAI)
        self.token_usage_log = TokenUsageLog()
        
    def chat(
        self,
        system_prompt: str,
        message: str,
        chat_history: Optional[List] = None,
    ) -> str:
        """
        Chat with the Gemini AI.
        
        Parameters
        ----------
        system_prompt : str
            The system prompt (instructions for the AI).
        message : str
            The user message.
        chat_history : Optional[List]
            Optional chat history.
            
        Returns
        -------
        str
            The model's response.
        """
        try:
            # Convert chat_history to Gemini format if provided
            gemini_history = self._convert_chat_history(chat_history) if chat_history else []
            
            # Add system prompt at the beginning if it's not empty
            if system_prompt:
                # Gemini doesn't have a dedicated system prompt, so add it as a user message
                gemini_history.append({
                    "role": "user",
                    "parts": [{"text": f"[System Instructions] {system_prompt}"}]
                })
                gemini_history.append({
                    "role": "model",
                    "parts": [{"text": "I'll follow these instructions."}]
                })
            
            # Add the current message
            gemini_history.append({
                "role": "user",
                "parts": [{"text": message}]
            })
            
            # Create a chat session
            if gemini_history:
                chat = self.model.start_chat(history=gemini_history[:-1])
                response = chat.send_message(message)
            else:
                response = self.model.generate_content(message)
                
            # Extract and return the text response
            text_response = response.text
                
            # Track token usage
            self.token_usage_log.log_request(
                prompt_tokens=len(message) // 4,  # Rough approximation, Gemini measures differently
                completion_tokens=len(text_response) // 4,
                model=self.model_name
            )
                
            return text_response
            
        except Exception as e:
            logger.error(f"Error in Gemini chat: {e}")
            return f"Error communicating with Gemini API: {str(e)}"
    
    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Complete a prompt using the Gemini AI.
        
        Parameters
        ----------
        prompt : str
            The prompt to complete.
        stop : Optional[List[str]]
            Optional list of stop sequences.
            
        Returns
        -------
        str
            The model's completion.
        """
        try:
            # Generate content
            response = self.model.generate_content(prompt)
            
            text_response = response.text
            
            # Apply stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in text_response:
                        text_response = text_response.split(stop_seq)[0]
            
            # Track token usage
            self.token_usage_log.log_request(
                prompt_tokens=len(prompt) // 4,  # Rough approximation
                completion_tokens=len(text_response) // 4,
                model=self.model_name
            )
            
            return text_response
            
        except Exception as e:
            logger.error(f"Error in Gemini completion: {e}")
            return f"Error with Gemini API: {str(e)}"
    
    def _convert_chat_history(self, chat_history: List) -> List:
        """
        Convert chat history to Gemini format.
        
        Parameters
        ----------
        chat_history : List
            The chat history in the internal format.
            
        Returns
        -------
        List
            The chat history in Gemini's format.
        """
        gemini_history = []
        
        for message in chat_history:
            if "role" in message and "content" in message:
                role = "user" if message["role"] == "user" else "model"
                gemini_history.append({
                    "role": role,
                    "parts": [{"text": message["content"]}]
                })
                
        return gemini_history


class TokenUsageLog:
    """Simple token usage tracker for Gemini."""
    
    def __init__(self):
        """Initialize the token usage log."""
        self.requests = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        
    def log_request(self, prompt_tokens: int, completion_tokens: int, model: str):
        """
        Log a request to the token usage log.
        
        Parameters
        ----------
        prompt_tokens : int
            The number of prompt tokens.
        completion_tokens : int
            The number of completion tokens.
        model : str
            The model name.
        """
        # Gemini price estimation (very rough approximation)
        if "2.0-flash" in model:
            prompt_cost = prompt_tokens * 0.0000007
            completion_cost = completion_tokens * 0.0000014
        elif "2.0-pro" in model:
            prompt_cost = prompt_tokens * 0.0000035
            completion_cost = completion_tokens * 0.000007
        else:
            prompt_cost = prompt_tokens * 0.0000007
            completion_cost = completion_tokens * 0.0000014
            
        total_cost = prompt_cost + completion_cost
        
        self.requests.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": total_cost,
        })
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += total_cost
        
    def total_tokens(self) -> int:
        """
        Get the total number of tokens used.
        
        Returns
        -------
        int
            The total number of tokens.
        """
        return self.total_prompt_tokens + self.total_completion_tokens
    
    def usage_cost(self) -> float:
        """
        Get the total cost of usage.
        
        Returns
        -------
        float
            The total cost.
        """
        return self.total_cost
    
    def is_openai_model(self) -> bool:
        """
        Check if this is an OpenAI model.
        
        Returns
        -------
        bool
            Always False for Gemini.
        """
        return False
