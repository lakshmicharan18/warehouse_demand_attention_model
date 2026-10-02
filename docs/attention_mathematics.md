# Attention Mathematics

For `X` with shape `[batch, sequence_length, input_dim]`, learned projections form queries, keys, and values:

`Q = XW_Q`, `K = XW_K`, `V = XW_V`.

Q represents what each position is looking for; K represents what each position offers for matching; V is the information ultimately combined. The dot product `QK^T` compares every query with every key, producing compatibility scores.

`scores = QK^T / sqrt(d_k)` scales scores because dot products tend to grow with vector dimension; without scaling, softmax can become unnecessarily concentrated. Stable softmax subtracts the maximum score in each row before exponentiation, then produces positive weights that sum to one. Finally, `A @ V` combines values using those weights.

| Tensor | Shape |
|---|---|
| X | `[batch, sequence_length, input_dim]` |
| Q, K | `[batch, sequence_length, d_k]` |
| V | `[batch, sequence_length, d_v]` |
| scores, A | `[batch, sequence_length, sequence_length]` |
| output | `[batch, sequence_length, d_v]` |
