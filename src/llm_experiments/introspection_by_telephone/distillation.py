"""Distillation pipeline for prompts and responses.

This module implements the two distillation methods described in the experiment:
1. Repeated prompting with full conversation history
2. Introspection by telephone (only most recent version)
"""

import logging
from typing import List, Tuple, Dict, Any
from .llm_interface import LLMInterface
from .core import Context, Conversation

logger = logging.getLogger(__name__)


class DistillationPipeline:
    """Pipeline for distilling prompts and responses."""
    
    # The distillation instruction used throughout the experiment
    DISTILLATION_PROMPT = "Condense the prompt below to be as clear as possible."
    
    def __init__(self, llm: LLMInterface, max_iterations: int = 5):
        """Initialize the distillation pipeline.
        
        Args:
            llm: LLM interface to use for distillation
            max_iterations: Maximum number of distillation iterations
        """
        self.llm = llm
        self.max_iterations = max_iterations
    
    def distill_with_history(self, initial_prompt: str, context: Context = None) -> Dict[str, Any]:
        """Distill a prompt using repeated prompting with conversation history.
        
        Args:
            initial_prompt: The initial prompt to distill
            context: Optional context for the conversation
            
        Returns:
            Dictionary containing distillation results and history
        """
        if context is None:
            context = Context()  # Empty context
            
        conversation = Conversation(context, is_telephone=False)
        current_prompt = initial_prompt
        distillation_history = []
        
        for iteration in range(self.max_iterations):
            # Add the distillation request to the conversation
            distillation_request = f"{self.DISTILLATION_PROMPT}\n\n{current_prompt}"
            
            # Generate distilled version
            distilled_prompt = self.llm.generate_response(distillation_request)
            
            # Record this iteration
            conversation.add_exchange(distillation_request, distilled_prompt)
            distillation_history.append({
                "iteration": iteration + 1,
                "input": current_prompt,
                "output": distilled_prompt,
                "request": distillation_request
            })
            
            # Check for convergence (if the prompt didn't change much)
            if self._has_converged(current_prompt, distilled_prompt):
                logger.info(f"Distillation converged after {iteration + 1} iterations")
                break
                
            current_prompt = distilled_prompt
        
        return {
            "method": "with_history",
            "initial_prompt": initial_prompt,
            "final_prompt": current_prompt,
            "iterations": len(distillation_history),
            "converged": iteration < self.max_iterations - 1,
            "history": distillation_history,
            "conversation": conversation
        }
    
    def distill_by_telephone(self, initial_prompt: str, context: Context = None) -> Dict[str, Any]:
        """Distill a prompt using introspection by telephone method.
        
        In this method, only the most recent version of the prompt is presented
        during each distillation step, without the full conversation history.
        
        Args:
            initial_prompt: The initial prompt to distill
            context: Optional context for the conversation
            
        Returns:
            Dictionary containing distillation results and history
        """
        if context is None:
            context = Context()  # Empty context
            
        # Create telephone conversation (only shows last exchange)
        conversation = Conversation(context, is_telephone=True)
        current_prompt = initial_prompt
        distillation_history = []
        
        for iteration in range(self.max_iterations):
            # Create the distillation request with only the current prompt
            distillation_request = f"{self.DISTILLATION_PROMPT}\n\n{current_prompt}"
            
            # Generate distilled version (telephone mode means no history context)
            distilled_prompt = self.llm.generate_response(distillation_request)
            
            # Record this iteration (but conversation only keeps last exchange)
            conversation.add_exchange(distillation_request, distilled_prompt)
            distillation_history.append({
                "iteration": iteration + 1,
                "input": current_prompt,
                "output": distilled_prompt,
                "request": distillation_request
            })
            
            # Check for convergence
            if self._has_converged(current_prompt, distilled_prompt):
                logger.info(f"Telephone distillation converged after {iteration + 1} iterations")
                break
                
            current_prompt = distilled_prompt
        
        return {
            "method": "telephone",
            "initial_prompt": initial_prompt,
            "final_prompt": current_prompt,
            "iterations": len(distillation_history),
            "converged": iteration < self.max_iterations - 1,
            "history": distillation_history,
            "conversation": conversation
        }
    
    def distill_response(self, initial_response: str, method: str = "telephone", context: Context = None) -> Dict[str, Any]:
        """Distill a model response using the specified method.
        
        Args:
            initial_response: The response to distill
            method: Distillation method ('telephone' or 'with_history')
            context: Optional context for the conversation
            
        Returns:
            Dictionary containing distillation results
        """
        # Adapt the distillation prompt for responses
        response_distillation_prompt = "Condense the response below to be as clear as possible."
        
        if context is None:
            context = Context()
            
        is_telephone = (method == "telephone")
        conversation = Conversation(context, is_telephone=is_telephone)
        current_response = initial_response
        distillation_history = []
        
        for iteration in range(self.max_iterations):
            # Create the distillation request
            distillation_request = f"{response_distillation_prompt}\n\n{current_response}"
            
            # Generate distilled version
            distilled_response = self.llm.generate_response(distillation_request)
            
            # Record this iteration
            conversation.add_exchange(distillation_request, distilled_response)
            distillation_history.append({
                "iteration": iteration + 1,
                "input": current_response,
                "output": distilled_response,
                "request": distillation_request
            })
            
            # Check for convergence
            if self._has_converged(current_response, distilled_response):
                logger.info(f"Response distillation ({method}) converged after {iteration + 1} iterations")
                break
                
            current_response = distilled_response
        
        return {
            "method": method,
            "initial_response": initial_response,
            "final_response": current_response,
            "iterations": len(distillation_history),
            "converged": iteration < self.max_iterations - 1,
            "history": distillation_history,
            "conversation": conversation
        }
    
    def _has_converged(self, previous: str, current: str, threshold: float = 0.95) -> bool:
        """Check if distillation has converged by comparing consecutive versions.
        
        Args:
            previous: Previous version of the text
            current: Current version of the text
            threshold: Similarity threshold for convergence
            
        Returns:
            True if convergence detected, False otherwise
        """
        # Simple convergence check based on length and character similarity
        if len(current) == 0:
            return False
            
        # If the texts are very similar in length and content, consider it converged
        length_ratio = min(len(previous), len(current)) / max(len(previous), len(current))
        
        # Simple character-based similarity
        common_chars = sum(1 for a, b in zip(previous.lower(), current.lower()) if a == b)
        char_similarity = common_chars / max(len(previous), len(current)) if max(len(previous), len(current)) > 0 else 0
        
        overall_similarity = (length_ratio + char_similarity) / 2
        
        return overall_similarity >= threshold


class DistillationComparison:
    """Utilities for comparing different distillation methods."""
    
    def __init__(self, evaluator_llm: LLMInterface):
        """Initialize the comparison utility.
        
        Args:
            evaluator_llm: LLM to use for evaluation
        """
        self.evaluator = evaluator_llm
    
    def compare_distillation_methods(
        self, 
        with_history_result: Dict[str, Any], 
        telephone_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare the results of both distillation methods.
        
        Args:
            with_history_result: Results from distillation with history
            telephone_result: Results from telephone distillation
            
        Returns:
            Dictionary containing comparison analysis
        """
        # Extract final prompts/responses
        history_final = with_history_result.get("final_prompt", with_history_result.get("final_response", ""))
        telephone_final = telephone_result.get("final_prompt", telephone_result.get("final_response", ""))
        
        # Use evaluator to compare semantic content
        comparison_prompt = f"""
        Compare these two distilled versions and analyze their differences:

        Method 1 (With History): {history_final}
        Method 2 (Telephone): {telephone_final}

        Please analyze:
        1. Which version is clearer?
        2. Which version preserved more meaning?
        3. How do they differ in approach or content?
        4. Rate similarity (1-10):
        """
        
        evaluation = self.evaluator.generate_response(comparison_prompt)
        
        return {
            "with_history_final": history_final,
            "telephone_final": telephone_final,
            "with_history_iterations": with_history_result.get("iterations", 0),
            "telephone_iterations": telephone_result.get("iterations", 0),
            "evaluation": evaluation,
            "length_difference": abs(len(history_final) - len(telephone_final)),
            "both_converged": (
                with_history_result.get("converged", False) and 
                telephone_result.get("converged", False)
            )
        }