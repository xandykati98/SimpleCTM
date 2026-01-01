What if at each tick we calculated the sum of all the activations of attention, if this value SumAttn is higher we increase the loss, incentivizing later ticks to have fewer attention actvations in total, as training progresses the CTM can have its max ticks reduced gradually. This is an interesting idea for encouraging attention sparsity and potentially enabling dynamic tick reduction! The concept is to add a regularization term that penalizes high total attention activations, especially in later ticks.

Key Considerations
Normalization: Attention weights from nn.MultiheadAttention are already softmax-normalized (sum to 1 per query), so summing them gives n_heads per tick. You might want to normalize by this.
Sparsity vs. Concentration: This encourages lower total attention, but you might instead want to encourage concentrated attention (high entropy penalty) rather than just low magnitude.
Evaluation: You'd need to update evaluate() to handle the new return signature, or add a flag to skip attention_sums during eval.



If LM still ends looking good we can train on RL to instruct-tune it on some small env for simple math problems 

test lm with muon

ctm autoencoder