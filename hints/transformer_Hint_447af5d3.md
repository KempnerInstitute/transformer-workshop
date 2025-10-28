
Take a moment to think about what your embedding layer is doing before we move on:

*   What does each row of the embedding matrix represent? How does the output shape of the embedding layer relate to the input shape?

*Each row of the embedding matrix (self.wte.weight) corresponds to a token in the vocabulary.
That row is a learned vector representation of that token, which the model will adjust during training so that tokens used in similar contexts end up with similar vectors. The embedding matrix has shape (vocab_size, d_model).
When we feed in a batch of token IDs shaped (batch_size, seq_len), the embedding layer looks up the appropriate rows for each token, producing an output tensor of shape (batch_size, seq_len, d_model).*

*  How does the embedding layer learn during training?

*The embedding weights are trainable parameters.
During the forward pass, token IDs are used to look up embeddings.
During backpropagation, gradients flow through those lookups and update the corresponding rows in the embedding matrix.*

* What kind of information do the embedding vectors capture as the model trains?

*As the model learns, embeddings start to capture statistical and semantic relationships between tokens — tokens that appear in similar contexts (e.g., “dog” and “cat”) move closer together in the embedding space.*
