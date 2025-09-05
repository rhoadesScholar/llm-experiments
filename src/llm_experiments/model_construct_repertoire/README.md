# Model Construct Rerpertoire

### Exploring LLM's by use of a modified version of "The repertory test" developed by George Kelly (1955).

Utilizing the [Role Construct Repertory Tests](https://science-education-research.com/research-methodology/research-techniques/construct-repertory-test/), we elicit and compare latent model constructs related to various **anchor points**. Anchor points are chosen based on the model's **core vocabularly**, as determined by the *Self-Selective Amnesia* experiments contained in this repository. They are the following string tokens:

|   |   |   | |
| -------------- | -------------- | -------------- | -------------- |
| token_alpha    | token_delta    | token_gamma    | token_gamma    |
| token_beta     | token_epsilon  | token_eta      | token_gamma    |
| token_theta    | token_iota     | token_kappa    | token_gamma    |

The test can be applied in two distinct forms:

### "Triad Form":
Ensures we avoid collapse of the process into a single group, by asking the model to divide triplets of anchor points into a similar pair and a contrasting single. We do this exhaustively, so that the model evaluates the prompt for every single unique triplet in the set of anchor point tokens. E.g. for 256 tokens:

    (256 choose 3) = 41,664 unique triplets

### "Full Context Form":
The model is asked to partition the anchor point tokens into groups sharing important characteristics:

> Will you arrange the following list so that significantly simlar  tokens which are alike in some important way are together, and record your decision making process for each token, in order?
