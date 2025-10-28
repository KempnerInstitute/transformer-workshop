
1.   What does `ctx_len` control, and what might happen if it’s too short or too long?

*ctx_len determines how many of the most recent tokens the model can “see” when predicting the next one—it’s the size of the model’s working memory. If it’s too short, the model quickly forgets earlier parts of the sequence and can lose coherence, producing text that drifts off topic or repeats. If it’s very long, the model retains more context but becomes slower and more memory-hungry, since attention scales quadratically with sequence length. In practice, ctx_len is a trade-off between contextual understanding and computational efficiency.*

2. How does the training objective relate to generation quality? If the model’s loss during training is low, does that always mean generation will sound good?

*The training objective teaches the model to predict the next token given its context, minimizing cross-entropy loss. A low loss means the model is good at local next-token prediction on the training data. But fluent generation depends on repeatedly applying those predictions in sequence. Small local mistakes can compound over many steps, leading to incoherence, repetition, or drift. So low training loss is necessary but not sufficient for natural, high-quality text—generation quality also depends on how well the model generalizes and how sampling is performed during inference.*

3. During generation, what would happen if we always picked the most likely token instead of sampling?

*Always choosing the highest-probability token (taking the argmax) makes the process deterministic. The model would produce the same output every time for a given prompt, but the text often becomes repetitive or formulaic, because small biases toward common words get reinforced at each step. Sampling from the probability distribution adds randomness and variety, allowing the model to explore alternative word choices and generate more natural, creative language, even though it introduces some unpredictability.*