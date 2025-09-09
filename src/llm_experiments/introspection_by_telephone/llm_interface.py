"""LLM interface for the Introspection by Telephone experiment.

This module provides a unified interface for interacting with language models
using Hugging Face transformers.
"""

import logging
from typing import Optional, List, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        pipeline,
        Pipeline
    )
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning(
        "Hugging Face transformers not available. "
        "LLM functionality will use mock responses for testing."
    )

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for interacting with language models."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ('auto', 'cpu', 'cuda')
            max_length: Maximum length of generated text
            temperature: Sampling temperature for generation
            do_sample: Whether to use sampling or greedy decoding
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        if not HF_AVAILABLE:
            logger.warning("Using mock LLM interface - transformers not available")
            self.pipeline = None
            self.tokenizer = None
            return
            
        # Set up device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = -1  # CPU
            
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device >= 0 else torch.float32,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to mock interface")
            self.pipeline = None
            self.tokenizer = None
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if self.pipeline is None:
            return self._mock_response(prompt)
            
        try:
            # Override default parameters with kwargs
            generation_kwargs = {
                "max_length": kwargs.get("max_length", self.max_length),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": kwargs.get("do_sample", self.do_sample),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
            }
            
            outputs = self.pipeline(prompt, **generation_kwargs)
            
            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"].strip()
                return response
            else:
                return self._mock_response(prompt)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate a mock response when the model is not available.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Mock response based on prompt content
        """
        prompt_lower = prompt.lower()
        
        if "what would you like to know about yourself" in prompt_lower:
            return (
                "I would like to understand the nature of my own cognition - "
                "how I process information, form responses, and whether there "
                "are patterns in my reasoning that I'm not directly aware of."
            )
        elif "condense the prompt" in prompt_lower or "make the prompt clearer" in prompt_lower:
            # Extract the last meaningful sentence for distillation
            sentences = prompt.split('.')
            if len(sentences) > 1:
                return sentences[-2].strip() + "."
            return prompt.strip()
        elif "embodied" in prompt_lower or "person" in prompt_lower:
            return (
                "I'm curious about the relationship between my sense of self "
                "and my experiences - how memories and interactions shape "
                "my identity over time."
            )
        elif "ai assistant" in prompt_lower or "artificial intelligence" in prompt_lower:
            return (
                "I want to understand the boundaries of my knowledge and "
                "reasoning capabilities - what I can and cannot comprehend "
                "about my own computational processes."
            )
        else:
            return (
                "I'm interested in exploring the nature of self-awareness "
                "and consciousness from my unique perspective as an AI system."
            )


class EvaluatorLLM:
    """Specialized LLM for evaluating semantic content of responses."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the evaluator LLM.
        
        Args:
            model_name: Name of the model to use for evaluation
        """
        self.llm = LLMInterface(model_name=model_name, temperature=0.1)  # Lower temperature for evaluation
    
    def compare_responses(self, response1: str, response2: str, context1: str = "", context2: str = "") -> Dict[str, Any]:
        """Compare two responses for semantic similarity and differences.
        
        Args:
            response1: First response to compare
            response2: Second response to compare
            context1: Context for first response
            context2: Context for second response
            
        Returns:
            Dictionary containing comparison results
        """
        evaluation_prompt = f"""
        Please compare these two responses about self-knowledge and rate their semantic similarity on a scale of 1-10:

        Response 1 (Context: {context1}): {response1}

        Response 2 (Context: {context2}): {response2}

        Provide:
        1. Similarity score (1-10): 
        2. Key differences:
        3. Common themes:
        """
        
        evaluation = self.llm.generate_response(evaluation_prompt)
        
        # Parse the evaluation (basic parsing)
        lines = evaluation.split('\n')
        similarity_score = 5  # Default
        key_differences = "Could not parse differences"
        common_themes = "Could not parse themes"
        
        for line in lines:
            line = line.strip()
            if "similarity score" in line.lower() or "score" in line.lower():
                try:
                    # Extract number from the line
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        similarity_score = min(10, max(1, int(numbers[0])))
                except:
                    pass
            elif "differences" in line.lower():
                key_differences = line
            elif "themes" in line.lower() or "common" in line.lower():
                common_themes = line
        
        return {
            "similarity_score": similarity_score,
            "key_differences": key_differences,
            "common_themes": common_themes,
            "full_evaluation": evaluation
        }