## Traditional ML Models

For these models, we need to go from tokenization, to vectorization _(e.g. FastText, BERT)_ to Aggregation.

### Tokenization

Just split the text of the template into tokens.

```
TEMPLATE => TOKENS
```

### Vectorization

For vectorization, we can use pre-trained models like FastText _(crawl-300d-2M.vec)_ or BERT _(bert-base-uncased)_.

```
TOKEN => VECTOR

logadu vectorize /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/Fox/drain/Fox_templates.csv /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/crawl-300d-2M.vec

```

### Aggregation

For aggregation, we can use simple methods like averaging or more complex methods like attention mechanisms.

```
TEMPLATE => TOKENS => VECTORS => AGGREGATED_VECTOR
```

### Consumption

PCA, RF, KNN and Autoencoder models will use the aggregated vector as input _(Mean Pooling)_.

```
AGGREGATED_VECTOR => PCA/RF/KNN/Autoencoder
```

LogRobust and NeauralLog will consume the sequence of vectors directly.

```
TEMPLATE => TOKENS => VECTORS => LogRobust/NeuralLog
```
