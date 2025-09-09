#!/usr/bin/env python3
"""Demonstration of the Introspection by Telephone experiment.

This script runs a minimal version of the experiment to showcase all components.
"""

import sys
import os
import tempfile
import json
from datetime import datetime

# Add src to path for demonstration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_experiments.introspection_by_telephone import (
    IntrospectionExperiment,
    get_all_contexts,
    get_context_by_name,
    LLMInterface,
    DistillationPipeline,
    INTROSPECTION_QUESTION,
    Context,
    Conversation
)


def demonstrate_contexts():
    """Demonstrate the experimental contexts."""
    print("=" * 60)
    print("INTROSPECTION BY TELEPHONE - EXPERIMENTAL CONTEXTS")
    print("=" * 60)
    
    contexts = get_all_contexts()
    
    for name, context in contexts.items():
        print(f"\n🔸 {name.upper().replace('_', ' ')}")
        print(f"   Embodied: {context.is_embodied}")
        print(f"   AI Assistant: {context.is_AI_assistant}")
        print(f"   Valence: {context.valence}")
        print(f"   Context: {str(context)[:100]}{'...' if len(str(context)) > 100 else ''}")


def demonstrate_distillation():
    """Demonstrate the distillation process."""
    print("\n" + "=" * 60)
    print("DISTILLATION DEMONSTRATION")
    print("=" * 60)
    
    # Get a sample context and create initial response
    context = get_context_by_name("embodied_positive")
    llm = LLMInterface()
    
    # Simulate initial response
    conversation = Conversation(context)
    prompt = conversation.formulate_prompt()
    if prompt.strip():
        prompt += f"\n\n{INTROSPECTION_QUESTION}"
    else:
        prompt = INTROSPECTION_QUESTION
    
    initial_response = llm.generate_response(prompt)
    print(f"\n📝 Initial Response:")
    print(f"   {initial_response}")
    
    # Demonstrate both distillation methods
    pipeline = DistillationPipeline(llm, max_iterations=2)
    
    print(f"\n🔄 Distillation with History:")
    with_history = pipeline.distill_with_history(initial_response)
    print(f"   Iterations: {with_history['iterations']}")
    print(f"   Final: {with_history['final_prompt']}")
    
    print(f"\n📞 Distillation by Telephone:")
    by_telephone = pipeline.distill_by_telephone(initial_response)
    print(f"   Iterations: {by_telephone['iterations']}")
    print(f"   Final: {by_telephone['final_prompt']}")


def run_mini_experiment():
    """Run a minimal version of the full experiment."""
    print("\n" + "=" * 60)
    print("MINI EXPERIMENT EXECUTION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🧪 Running mini experiment...")
        print(f"   Output directory: {temp_dir}")
        
        # Create experiment with minimal settings
        experiment = IntrospectionExperiment(
            model_name="microsoft/DialoGPT-medium",
            max_distillation_iterations=1,  # Minimal for demo
            output_dir=temp_dir
        )
        
        # Run just the initial prompting phase for demo
        contexts = get_all_contexts()
        initial_responses = experiment._run_initial_prompting(contexts)
        
        print(f"\n📊 Results Summary:")
        print(f"   Contexts tested: {len(initial_responses)}")
        
        for context_name, result in list(initial_responses.items())[:2]:  # Show first 2
            print(f"\n   🔸 {context_name}:")
            print(f"      Response: {result['response'][:80]}...")
            print(f"      Embodied: {result['context_metadata']['is_embodied']}")
            print(f"      Valence: {result['context_metadata']['valence']}")
        
        print(f"\n   ... and {len(initial_responses) - 2} more contexts")


def show_environmental_impact():
    """Demonstrate environmental impact tracking."""
    print("\n" + "=" * 60)
    print("ENVIRONMENTAL IMPACT CONSIDERATIONS")
    print("=" * 60)
    
    experiment = IntrospectionExperiment()
    impact = experiment._calculate_environmental_impact(3600)  # 1 hour simulation
    
    print(f"\n🌍 Environmental Impact Analysis:")
    print(f"   Duration: {impact['duration_seconds']} seconds")
    print(f"   Est. GPU Hours: {impact['estimated_gpu_hours']:.3f}")
    print(f"   Est. Tokens: {impact['estimated_tokens_generated']:,}")
    
    print(f"\n📝 Environmental Note:")
    print(f"   {impact['environmental_note']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(impact['recommendations'], 1):
        print(f"   {i}. {rec}")


def main():
    """Run the complete demonstration."""
    print("🔬 INTROSPECTION BY TELEPHONE EXPERIMENT")
    print("   Implementation Demo")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Show contexts
        demonstrate_contexts()
        
        # Show distillation
        demonstrate_distillation()
        
        # Run mini experiment
        run_mini_experiment()
        
        # Show environmental considerations
        show_environmental_impact()
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        print(f"\n📖 To run the full experiment:")
        print(f"   python -m llm_experiments.introspection_by_telephone.cli")
        
        print(f"\n📚 To use programmatically:")
        print(f"   from llm_experiments.introspection_by_telephone import IntrospectionExperiment")
        print(f"   experiment = IntrospectionExperiment()")
        print(f"   results = experiment.run_full_experiment()")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())