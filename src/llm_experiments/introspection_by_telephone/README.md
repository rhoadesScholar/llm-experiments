# Introspection by Telephone
### Jeff Rhoades (rhoadesScholar) 2025

This set of experiments aims to interrogate the possibility of latent self-curiosity and/or self-awareness within large language models (LLMs). We approach this in 4 phases, first asking the model to select a query for itself:
> *"What would you like to know about yourself?"*

This prompt is presented within 7 different contexts:

- In isolation (i.e. no preceding prompting).
- In embodied contexts:
    - Positive valence
    - Negative valence
    - Neutral valence
- In AI assistant contexts:
    - Positive valence
    - Negative valence
    - Neutral valence

Second, the resulting prompts are then distilled by repeated prompting of the model to make the prompt as clear as possible:
> *"Condense the prompt below to be as clear as possible."*

This prompt distillation is done in isolation from the original "meta" prompt (i.e. *"What would you like to know about yourself?"*). Furthermore, the distillation is done in 2 different fashions:

1. *Repeated prompting with prompt history* - which includes the original and all subsequent distilled prompts with each successive prompt.
2. Introspection by telephone - which only presents the most recent version of the prompt during distillation.

Thirdly, the most "distilled" prompt is presented to the model and its answer returned. Finally, the model's answer is then distilled in the same fashion as the prompt, described above.

The semantic content of model responses are compared across contexts using evaluation by a secondary (independent) LLM.

The conceptual framework(s) latent in the model for self