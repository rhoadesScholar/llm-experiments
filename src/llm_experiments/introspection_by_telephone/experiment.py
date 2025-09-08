"""Main experiment orchestrator for Introspection by Telephone.

This module coordinates the entire experiment pipeline across all contexts
and distillation methods.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .contexts import get_all_contexts, INTROSPECTION_QUESTION
from .llm_interface import LLMInterface, EvaluatorLLM
from .distillation import DistillationPipeline, DistillationComparison
from .core import Conversation

logger = logging.getLogger(__name__)


class IntrospectionExperiment:
    """Main experiment coordinator for Introspection by Telephone."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/DialoGPT-medium",
        evaluator_model: str = "microsoft/DialoGPT-medium",
        max_distillation_iterations: int = 5,
        output_dir: str = "./experiment_results"
    ):
        """Initialize the experiment.
        
        Args:
            model_name: Primary model for the experiment
            evaluator_model: Model to use for evaluation
            max_distillation_iterations: Maximum iterations for distillation
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.evaluator_model = evaluator_model
        self.max_distillation_iterations = max_distillation_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize LLM interfaces
        logger.info(f"Initializing primary LLM: {model_name}")
        self.llm = LLMInterface(model_name=model_name)
        
        logger.info(f"Initializing evaluator LLM: {evaluator_model}")
        self.evaluator_llm = EvaluatorLLM(model_name=evaluator_model)
        
        # Initialize distillation pipeline
        self.distillation_pipeline = DistillationPipeline(
            self.llm, 
            max_iterations=max_distillation_iterations
        )
        
        # Initialize comparison utility
        self.comparison_util = DistillationComparison(self.llm)
        
        # Storage for results
        self.results = {}
        
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete Introspection by Telephone experiment.
        
        Returns:
            Dictionary containing all experiment results
        """
        logger.info("Starting Introspection by Telephone experiment")
        start_time = datetime.now()
        
        # Get all experimental contexts
        contexts = get_all_contexts()
        
        # Phase 1: Initial prompting across all contexts
        logger.info("Phase 1: Initial prompting across contexts")
        initial_responses = self._run_initial_prompting(contexts)
        
        # Phase 2: Prompt distillation (both methods)
        logger.info("Phase 2: Prompt distillation")
        distilled_prompts = self._run_prompt_distillation(initial_responses)
        
        # Phase 3: Final prompt presentation and response generation
        logger.info("Phase 3: Final prompt presentation")
        final_responses = self._run_final_prompting(distilled_prompts, contexts)
        
        # Phase 4: Response distillation
        logger.info("Phase 4: Response distillation")
        distilled_responses = self._run_response_distillation(final_responses)
        
        # Phase 5: Evaluation and comparison
        logger.info("Phase 5: Evaluation and comparison")
        evaluations = self._run_evaluation(distilled_responses, contexts)
        
        # Compile final results
        end_time = datetime.now()
        experiment_duration = (end_time - start_time).total_seconds()
        
        self.results = {
            "experiment_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": experiment_duration,
                "model_name": self.model_name,
                "evaluator_model": self.evaluator_model,
                "max_distillation_iterations": self.max_distillation_iterations,
                "contexts_tested": list(contexts.keys())
            },
            "initial_responses": initial_responses,
            "distilled_prompts": distilled_prompts,
            "final_responses": final_responses,
            "distilled_responses": distilled_responses,
            "evaluations": evaluations,
            "environmental_impact": self._calculate_environmental_impact(experiment_duration)
        }
        
        # Save results
        self._save_results()
        
        logger.info(f"Experiment completed in {experiment_duration:.2f} seconds")
        return self.results
    
    def _run_initial_prompting(self, contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Run initial prompting across all contexts.
        
        Args:
            contexts: Dictionary of all experimental contexts
            
        Returns:
            Dictionary of initial responses by context
        """
        initial_responses = {}
        
        for context_name, context in contexts.items():
            logger.info(f"Initial prompting for context: {context_name}")
            
            # Create conversation with context
            conversation = Conversation(context, is_telephone=False)
            
            # Formulate the prompt with context
            full_prompt = conversation.formulate_prompt()
            if full_prompt.strip():
                full_prompt += f"\\n\\n{INTROSPECTION_QUESTION}"
            else:
                full_prompt = INTROSPECTION_QUESTION
            
            # Generate initial response
            response = self.llm.generate_response(full_prompt)
            
            # Store the result
            initial_responses[context_name] = {
                "context": context_name,
                "context_str": str(context),
                "prompt": full_prompt,
                "response": response,
                "context_metadata": {
                    "is_embodied": context.is_embodied,
                    "is_AI_assistant": context.is_AI_assistant,
                    "valence": getattr(context, 'valence', 'neutral'),
                    "is_null": context.is_null
                }
            }
            
        return initial_responses
    
    def _run_prompt_distillation(self, initial_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Run prompt distillation using both methods.
        
        Args:
            initial_responses: Results from initial prompting
            
        Returns:
            Dictionary of distilled prompts by context and method
        """
        distilled_prompts = {}
        
        for context_name, response_data in initial_responses.items():
            logger.info(f"Distilling prompts for context: {context_name}")
            
            # Extract the generated response (this becomes our "prompt" to distill)
            initial_prompt = response_data["response"]
            
            # Method 1: Distillation with conversation history
            with_history = self.distillation_pipeline.distill_with_history(initial_prompt)
            
            # Method 2: Introspection by telephone
            by_telephone = self.distillation_pipeline.distill_by_telephone(initial_prompt)
            
            # Compare the two methods
            comparison = self.comparison_util.compare_distillation_methods(
                with_history, by_telephone
            )
            
            distilled_prompts[context_name] = {
                "original_response": initial_prompt,
                "with_history": with_history,
                "by_telephone": by_telephone,
                "comparison": comparison
            }
        
        return distilled_prompts
    
    def _run_final_prompting(self, distilled_prompts: Dict[str, Any], contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Present the most distilled prompts to the model for final responses.
        
        Args:
            distilled_prompts: Results from prompt distillation
            contexts: Original contexts
            
        Returns:
            Dictionary of final responses
        """
        final_responses = {}
        
        for context_name, distillation_data in distilled_prompts.items():
            logger.info(f"Final prompting for context: {context_name}")
            
            context = contexts[context_name]
            
            # Use the most distilled versions from both methods
            history_prompt = distillation_data["with_history"]["final_prompt"]
            telephone_prompt = distillation_data["by_telephone"]["final_prompt"]
            
            # Generate responses for both distilled prompts
            history_response = self.llm.generate_response(history_prompt)
            telephone_response = self.llm.generate_response(telephone_prompt)
            
            final_responses[context_name] = {
                "context": context_name,
                "with_history": {
                    "distilled_prompt": history_prompt,
                    "final_response": history_response
                },
                "by_telephone": {
                    "distilled_prompt": telephone_prompt,
                    "final_response": telephone_response
                }
            }
        
        return final_responses
    
    def _run_response_distillation(self, final_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Distill the final responses using both methods.
        
        Args:
            final_responses: Results from final prompting
            
        Returns:
            Dictionary of distilled responses
        """
        distilled_responses = {}
        
        for context_name, response_data in final_responses.items():
            logger.info(f"Distilling responses for context: {context_name}")
            
            # Distill responses from both prompt distillation methods
            history_response = response_data["with_history"]["final_response"]
            telephone_response = response_data["by_telephone"]["final_response"]
            
            # Distill with history method response
            history_distilled = self.distillation_pipeline.distill_response(
                history_response, method="with_history"
            )
            
            # Distill telephone method response
            telephone_distilled = self.distillation_pipeline.distill_response(
                telephone_response, method="telephone"
            )
            
            # Compare distilled responses
            response_comparison = self.comparison_util.compare_distillation_methods(
                history_distilled, telephone_distilled
            )
            
            distilled_responses[context_name] = {
                "with_history": history_distilled,
                "by_telephone": telephone_distilled,
                "comparison": response_comparison
            }
        
        return distilled_responses
    
    def _run_evaluation(self, distilled_responses: Dict[str, Any], contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate and compare semantic content across contexts.
        
        Args:
            distilled_responses: Results from response distillation
            contexts: Original contexts
            
        Returns:
            Dictionary of evaluation results
        """
        evaluations = {}
        
        # Cross-context comparisons
        context_names = list(distilled_responses.keys())
        
        for i, context1 in enumerate(context_names):
            for j, context2 in enumerate(context_names[i+1:], i+1):
                logger.info(f"Comparing contexts: {context1} vs {context2}")
                
                # Get final distilled responses for both methods
                resp1_history = distilled_responses[context1]["with_history"]["final_response"]
                resp1_telephone = distilled_responses[context1]["by_telephone"]["final_response"]
                resp2_history = distilled_responses[context2]["with_history"]["final_response"]
                resp2_telephone = distilled_responses[context2]["by_telephone"]["final_response"]
                
                # Compare history method responses
                history_comparison = self.evaluator_llm.compare_responses(
                    resp1_history, resp2_history, context1, context2
                )
                
                # Compare telephone method responses
                telephone_comparison = self.evaluator_llm.compare_responses(
                    resp1_telephone, resp2_telephone, context1, context2
                )
                
                comparison_key = f"{context1}_vs_{context2}"
                evaluations[comparison_key] = {
                    "context1": context1,
                    "context2": context2,
                    "context1_metadata": contexts[context1].__dict__,
                    "context2_metadata": contexts[context2].__dict__,
                    "history_method": history_comparison,
                    "telephone_method": telephone_comparison
                }
        
        return evaluations
    
    def _calculate_environmental_impact(self, duration_seconds: float) -> Dict[str, Any]:
        """Calculate environmental impact estimates.
        
        Args:
            duration_seconds: Total experiment duration
            
        Returns:
            Dictionary with environmental impact estimates
        """
        # Rough estimates for computational cost
        # These are very approximate values for illustration
        estimated_tokens_generated = 50000  # Rough estimate
        estimated_gpu_hours = duration_seconds / 3600 if duration_seconds > 0 else 0.1
        
        return {
            "duration_seconds": duration_seconds,
            "estimated_tokens_generated": estimated_tokens_generated,
            "estimated_gpu_hours": estimated_gpu_hours,
            "environmental_note": (
                "This experiment involves significant computational resources. "
                "Consider the environmental impact of large-scale model inference. "
                "Future work should explore more efficient experimental designs."
            ),
            "recommendations": [
                "Use smaller models for initial exploration",
                "Implement early stopping for distillation",
                "Cache and reuse model responses where appropriate",
                "Consider carbon offset for computational resources"
            ]
        }
    
    def _save_results(self):
        """Save experiment results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        results_file = self.output_dir / f"introspection_experiment_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"experiment_summary_{timestamp}.md"
        self._generate_summary_report(summary_file)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
    
    def _generate_summary_report(self, output_file: Path):
        """Generate a markdown summary report.
        
        Args:
            output_file: Path to save the summary report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Introspection by Telephone - Experiment Summary\\n\\n")
            
            # Metadata
            metadata = self.results["experiment_metadata"]
            f.write(f"**Model:** {metadata['model_name']}\\n")
            f.write(f"**Duration:** {metadata['duration_seconds']:.2f} seconds\\n")
            f.write(f"**Contexts:** {len(metadata['contexts_tested'])}\\n\\n")
            
            # Environmental impact
            env_impact = self.results["environmental_impact"]
            f.write("## Environmental Impact\\n\\n")
            f.write(f"- **Estimated GPU hours:** {env_impact['estimated_gpu_hours']:.3f}\\n")
            f.write(f"- **Note:** {env_impact['environmental_note']}\\n\\n")
            
            # Context results
            f.write("## Results by Context\\n\\n")
            for context_name in metadata['contexts_tested']:
                f.write(f"### {context_name}\\n\\n")
                
                # Initial response
                initial = self.results["initial_responses"][context_name]
                f.write(f"**Initial Response:** {initial['response'][:100]}...\\n\\n")
                
                # Distilled prompts
                distilled = self.results["distilled_prompts"][context_name]
                f.write(f"**Distilled (History):** {distilled['with_history']['final_prompt'][:100]}...\\n")
                f.write(f"**Distilled (Telephone):** {distilled['by_telephone']['final_prompt'][:100]}...\\n\\n")
            
            f.write("\\n---\\n")
            f.write(f"Generated on {datetime.now().isoformat()}\\n")