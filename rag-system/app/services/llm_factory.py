"""
LLM Factory: Manages OpenAI client with structured outputs via instructor.
Instructor forces the LLM to return validated Pydantic models.
"""
from typing import Any, Dict, List, Type
import instructor
from openai import OpenAI
from pydantic import BaseModel
from app.config.settings import get_settings

class LLMFactory:
    """Factory for creating LLM clients with structured output support."""
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM client.
        
        Args:
            provider: Currently only "openai" supported
        """
        if provider != "openai":
            raise ValueError(f"Only 'openai' provider supported, got: {provider}")
        
        self.provider = provider
        self.settings = get_settings().openai
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> Any:
        """
        Create instructor-wrapped OpenAI client.
        Instructor patches the client to validate responses against Pydantic schemas.
        """
        return instructor.from_openai(
            OpenAI(api_key=self.settings.api_key)
        )
    
    def create_completion(
        self, 
        response_model: Type[BaseModel], 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> BaseModel:
        """Generate structured completion."""
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "response_model": response_model,
            "messages": messages,
        }
        
        return self.client.chat.completions.create(**completion_params)