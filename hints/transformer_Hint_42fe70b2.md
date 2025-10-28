

*   Why does the model need position embeddings in addition to token embeddings?

*Token embeddings tell the model what each token is, but not where it appears in the sequence. Without position information, the model would treat text as a “bag of words,” unable to distinguish between different word orders (e.g., “dog bites man” vs. “man bites dog”). So, position embeddings give the model a sense of order and structure, which is essential for understanding sequences.*

*   Why are we adding the token and position embeddings together, instead of concatenating them?

*Both token and position embeddings have the same dimensionality (d_model), meaning each represents information in the same feature space. By adding them, we combine what the token is (its identity) and where it is (its position) into a single vector of the same size. This keeps the total embedding dimension fixed — so the next layers of the model (attention, feedforward, etc.) can process the combined information without any change in shape.*