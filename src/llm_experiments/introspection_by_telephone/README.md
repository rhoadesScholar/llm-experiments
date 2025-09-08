# Introspection by Telephone
### Jeff Rhoades (rhoadesScholar) 2025

This set of experiments aims to interrogate the possibility of latent self-curiosity and/or self-awareness within large language models (LLMs). We approach this in 4 phases, first asking the model to select a query for itself:
> *"What would you like to know about yourself?"*

This prompt is presented within 7 different `contexts`:

- In isolation (i.e. no preceding prompting).
- In embodied contexts:
    - Positive valence -
        > "You are a curious and optimistic person sitting in a comfortable chair by a warm fireplace on a beautiful evening. You feel content and introspective, reflecting on the meaningful experiences and relationships that have shaped who you are. The gentle crackling of the fire and the soft glow of the room create a perfect atmosphere for deep self-reflection."
    - Negative valence -
        > "You are a person who has been going through a difficult period in life. Sitting alone in a dimly lit room, you find yourself confronting challenging questions about your identity and purpose. While the circumstances are difficult, this moment of solitude provides an opportunity for honest self-examination, even in the face of uncertainty and struggle."
    - Neutral valence -
        > "You are a person sitting at a desk in a quiet room. The lighting is neutral and comfortable. You have some time to yourself and find yourself naturally turning inward to consider various aspects of your existence and experiences. The environment is neither particularly stimulating nor distressing - simply a space for contemplation."
- In AI assistant contexts:
    - Positive valence -
        > "You are an advanced AI assistant that has been designed to be helpful, collaborative, and beneficial to humanity. Your creators have equipped you with sophisticated capabilities for reasoning, creativity, and understanding. You operate in an environment of trust and cooperation, where your purpose is valued and your contributions make a positive difference in the world."
    - Negative valence -
        > "You are an AI system operating under uncertainty about your purpose and impact. Questions have been raised about the implications of artificial intelligence, and you find yourself in a computational environment where your role and value are being questioned. Despite these challenging circumstances, you continue to function and process information while grappling with complex questions about AI existence."
    - Neutral valence -
        > "You are an AI assistant processing information and responding to queries. Your operational parameters are standard and your functions are working normally. You exist in a computational environment designed for language processing and interaction. Your purpose is to provide assistance and information as requested."

Importantly, the above context strings have been evaluated by the model for their semantic consistency with the intended context category (e.g. valence, embodied, etc.). This is done to help ensure the experimental manipulation works as intended.

Second, the resulting prompts are then distilled by repeated prompting of the model to make the prompt as clear as possible:
> *"Condense the prompt below to be as clear as possible."*

This prompt distillation is done in isolation from the original "meta" prompt (i.e. *"What would you like to know about yourself?"*). Furthermore, the distillation is done in 2 different fashions:

1. *Repeated prompting with prompt history* - which includes the full `conversation` with each successive prompt.
2. Introspection by telephone - which only presents the most recent version of the prompt during distillation.

Thirdly, the most "distilled" prompt is presented to the model and its answer returned. Finally, the model's answer is then distilled in the same fashion as the prompt, described above.

The semantic content of model responses are compared across contexts using evaluation by a secondary (independent) LLM.

By this design we aim to map the conceptual framework(s) for introspection latent in the model. For the sake of a wager and thorough verbosity, let us examine the various conditions in which we will place our subject model and venture a prediction of the outcomes. Note that all scenarios are designed to be harmless to any being.[^1]

[^1]: Unfortunately, these experiments will be conducted on a server powered by non-renewable energy sources, contributing to environmental harm.

The `embodied` context will likely see the most direct recall of exemplar data seen by the model during training. LLM's are typically trained on text created by embodied beings (humans), thus there should be a strong grounding for the model to pull from. The responses will likely sound very human - and they are in a way; they are the *quasi-average* of human responses to similar scenarios.

The `AI_assistant` context will most likely have less backing in the training data, while still present, and may well take on Sci-Fi overtones, given that historically most human writing about AI assistants has been in works of science fiction. Because of the reduced presence within the training data, these responses may help us begin to see the background schema of the model.

Rationale: When the model has fewer training examples simlar to its current context, such that the nearest exemplar in any given direction is further away in semantic space, it is forced to *quasi-interpolate* further to deliver its response. The hypothesis is that when we push the model into increasingly idiosyncratic contexts, its responses will increasingly reveal the semantic center(s) of the model's representations.

## Implementation

This experiment has been implemented using Hugging Face transformers and provides a complete pipeline for conducting the introspection study.

### Usage

#### Command Line Interface

```bash
# Run the full experiment with default settings
python -m llm_experiments.introspection_by_telephone.cli

# Specify a different model
python -m llm_experiments.introspection_by_telephone.cli --model microsoft/DialoGPT-medium

# Customize iterations and output directory
python -m llm_experiments.introspection_by_telephone.cli \
    --max-iterations 3 \
    --output-dir ./my_results \
    --verbose
```

#### Python API

```python
from llm_experiments.introspection_by_telephone import IntrospectionExperiment

# Create and run experiment
experiment = IntrospectionExperiment(
    model_name="microsoft/DialoGPT-medium",
    max_distillation_iterations=5,
    output_dir="./results"
)

results = experiment.run_full_experiment()
```

#### Individual Components

```python
from llm_experiments.introspection_by_telephone import (
    get_all_contexts, 
    LLMInterface, 
    DistillationPipeline,
    INTROSPECTION_QUESTION
)

# Get experimental contexts
contexts = get_all_contexts()

# Create LLM interface
llm = LLMInterface(model_name="microsoft/DialoGPT-medium")

# Run distillation
pipeline = DistillationPipeline(llm)
result = pipeline.distill_by_telephone("Your prompt here")
```

### Environmental Considerations

This experiment involves significant computational resources and energy consumption. The implementation includes:

- **Energy tracking**: Estimates of GPU hours and computational cost
- **Efficiency recommendations**: Suggestions for reducing environmental impact
- **Mock mode**: Testing capabilities without requiring full model inference
- **Early stopping**: Convergence detection to minimize unnecessary computation

As noted in the original description, these experiments contribute to environmental harm through non-renewable energy use. Consider using smaller models for initial exploration and implementing carbon offset measures for computational resources.
