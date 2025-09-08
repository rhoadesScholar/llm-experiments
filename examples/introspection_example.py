"""Example usage of the Introspection by Telephone experiment.

This script demonstrates how to use the experiment framework.
"""

import logging
from llm_experiments.introspection_by_telephone import (
    IntrospectionExperiment,
    get_all_contexts,
    INTROSPECTION_QUESTION
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_quick_example():
    """Run a quick example with minimal setup."""
    logger.info("Running Introspection by Telephone example")
    
    # Show available contexts
    contexts = get_all_contexts()
    logger.info(f"Available contexts: {list(contexts.keys())}")
    
    # Create experiment with minimal settings
    experiment = IntrospectionExperiment(
        model_name="microsoft/DialoGPT-medium",  # Small model for demo
        max_distillation_iterations=2,  # Fewer iterations for speed
        output_dir="./example_results"
    )
    
    # Run the full experiment
    results = experiment.run_full_experiment()
    
    # Print summary
    logger.info("Experiment completed!")
    logger.info(f"Duration: {results['experiment_metadata']['duration_seconds']:.2f} seconds")
    logger.info(f"Results saved to: ./example_results/")
    
    return results


def demonstrate_individual_components():
    """Demonstrate using individual components."""
    from llm_experiments.introspection_by_telephone import (
        Context, Conversation, LLMInterface, DistillationPipeline
    )
    
    logger.info("Demonstrating individual components")
    
    # 1. Create a context
    context = Context(
        context_str="You are reflecting on your experiences in a quiet moment.",
        embodied=True,
        valence="neutral",
        name="custom_context"
    )
    logger.info(f"Created context: {context}")
    
    # 2. Create LLM interface
    llm = LLMInterface()
    
    # 3. Create conversation and get initial response
    conversation = Conversation(context)
    prompt = conversation.formulate_prompt() + f"\n\n{INTROSPECTION_QUESTION}"
    
    response = llm.generate_response(prompt)
    logger.info(f"Initial response: {response}")
    
    # 4. Demonstrate distillation
    distillation = DistillationPipeline(llm, max_iterations=2)
    
    # Distill with history
    with_history = distillation.distill_with_history(response)
    logger.info(f"Distilled with history: {with_history['final_prompt']}")
    
    # Distill by telephone
    by_telephone = distillation.distill_by_telephone(response)
    logger.info(f"Distilled by telephone: {by_telephone['final_prompt']}")


if __name__ == "__main__":
    print("Introspection by Telephone - Example Usage")
    print("=" * 50)
    
    try:
        # Run quick example
        run_quick_example()
        
        print("\n" + "=" * 50)
        
        # Demonstrate components
        demonstrate_individual_components()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()