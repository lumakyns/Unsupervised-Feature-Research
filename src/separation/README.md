# Separation Pretraining

This experiment tests whether a CNN learns faster after its early filters are
nudged to behave less alike.

During separation pretraining, a batch of images is passed through a selected
convolution layer. For two randomly chosen filters in that layer, each filter's
outputs across the whole batch and all spatial positions are flattened into one
activation-response vector. If the two filters respond in similar places, the
vectors point in a similar direction; if they respond differently, their cosine
similarity is closer to zero.

The separation loss is:

```text
cosine_similarity(response_a, response_b)^2
```

Squaring the cosine similarity asks the responses to be different without asking
one filter to be the negative of the other. The loss is averaged over random
filter pairs and over the selected layers.

The config controls how much pretraining to use:

- `baseline`: no separation pretraining
- `layer1`: first convolution only
- `layer12`: first and second convolutions
- `layer123`: first, second, and third convolutions

After that, the model trains normally with cross-entropy. The separation loss
and regular loss are never mixed.

The experiment uses the prepared `.pt` splits under `data/mnist` and
`data/cifar10`. Create them with:

```bash
python data/download.py
```

Run from the repository root:

```bash
python -m src.separation.training --config src/separation/config.yaml
```

W&B sweeps can override the YAML values, especially `dataset` and `variant`, to
compare MNIST and CIFAR-10 across the different pretraining depths.
