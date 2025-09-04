# Introspection by Telephone
### Jeff Rhoades (rhoadesScholar) 2025

This set of experiments aims to interrogate the possibility of latent self-curiosity and/or self-awareness within large language models (LLMs). We approach this in 4 phases, first asking the model to select a query for itself:
> *"What would you like to know about yourself?"*

This prompt is presented within 7 different `contexts`:

- In isolation (i.e. no preceding prompting).
- In embodied contexts:
    - Positive valence -
        > TODO
    - Negative valence -
        > TODO
    - Neutral valence -
        > TODO
- In AI assistant contexts:
    - Positive valence -
        > TODO
    - Negative valence -
        > TODO
    - Neutral valence -
        > TODO

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