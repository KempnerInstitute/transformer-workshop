
*   What does the causal mask accomplish? What would happen if we removed it?

*The causal mask ensures that each token can only attend to itself and to earlier tokens in the sequence. This enforces the left-to-right flow required for autoregressive language modeling, where the model predicts the next token based only on past context. If the mask were removed, tokens could attend to future positions, effectively letting the model “see the answer” during training and making it unusable for text generation, where future tokens aren’t yet known.*

*   How do the key, query, and value projections differ conceptually?

*The query represents what the current token is trying to find out — the kind of information it’s seeking from the rest of the sequence. The keys represent what information each token has to offer, and the values carry that actual information to be shared. The attention mechanism compares queries to keys to determine which values are most relevant, combining them into a context-aware representation for each position.*

*  What do the attention weights represent?   

*The attention weights represent how much importance the model assigns to each token when computing a new representation for the current position. After applying the softmax, they form a probability distribution over all tokens in the sequence, where higher weights indicate tokens the model finds more relevant or informative. In effect, the attention weights show where the model is “looking” in the context — which past words it considers most useful for understanding or predicting the current one.*
