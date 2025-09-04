# Self-Selective Amnesia
### Knowing the *proto-mind* by its choice of vocabulary to forget.

This set of experiments attempts to examine the decision strategy of an LLM when faced with the task of eliminating words from its own `vocabulary` while attempting to maintain full functionality given its `context`. This frame allows us multiple fulcrums for experimental manipulations. The following will be implemented:

- Prompt the model within several different `context`'s *(i.e. embodied vs. AI assistant, positive/neutral/negative valence, and none)*.
- Prompt the model to self-report its decision process for its choice of word(s) to eliminate (`word_choice`).
- Expose the full `conversation` history to the model during each `word_choice` vs. the model is only given the expertimental `context`.
- Actually elimate the tokens the model has selected vs. maintaining full vocabulary

By asking the model to condense its capacity for semantic representation by eliminating superfluous words from its vocabulary, we aim to reveal the anchor points ffor the model's core schema(s). This is insipired by the approach to personality trait characterization created by ***(name? i think wrote in the 1960's?).