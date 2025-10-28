
For position embeddings, you can also use nn.Embedding. Instead of the first dimension being equal to vocab size, it should be equal to the context length (so you learn an embedding for each position in a sequence)
