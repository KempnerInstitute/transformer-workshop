
The output of the token embeddings forward pass has shape (batch size x context length x model dimension).

For the forward pass of the position embeddings, you only need to create a matrix of shape (context length by model dimension) because nothing depends on the actual data in each batch. Broadcasting will ensure you can still add this matrix to the token embeddings.