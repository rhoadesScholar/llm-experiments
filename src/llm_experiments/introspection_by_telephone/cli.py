#!/usr/bin/env python3
"""Command-line interface for running the Introspection by Telephone experiment."""

import argparse
import logging
import sys
from pathlib import Path

from llm_experiments.introspection_by_telephone import IntrospectionExperiment


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Introspection by Telephone experiment"
    )
    
    parser.add_argument(
        "--model", 
        default="microsoft/DialoGPT-medium",
        help="Primary model for the experiment (default: microsoft/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--evaluator-model",
        default="microsoft/DialoGPT-medium", 
        help="Model for evaluation (default: microsoft/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum distillation iterations (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./experiment_results",
        help="Output directory for results (default: ./experiment_results)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment parameters without running"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Show experiment configuration
    logger.info("Introspection by Telephone Experiment")
    logger.info("=" * 50)
    logger.info(f"Primary model: {args.model}")
    logger.info(f"Evaluator model: {args.evaluator_model}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 50)
    
    if args.dry_run:
        logger.info("Dry run mode - experiment parameters shown above")
        return 0
    
    try:
        # Initialize experiment
        experiment = IntrospectionExperiment(
            model_name=args.model,
            evaluator_model=args.evaluator_model,
            max_distillation_iterations=args.max_iterations,
            output_dir=args.output_dir
        )
        
        # Run the experiment
        results = experiment.run_full_experiment()
        
        # Print summary
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print brief summary
        metadata = results["experiment_metadata"]
        logger.info(f"Duration: {metadata['duration_seconds']:.2f} seconds")
        logger.info(f"Contexts tested: {len(metadata['contexts_tested'])}")
        
        env_impact = results["environmental_impact"]
        logger.info(f"Estimated GPU hours: {env_impact['estimated_gpu_hours']:.3f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())