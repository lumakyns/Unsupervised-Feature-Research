# Separation Pretraining

This experiment tests whether a CNN learns faster after its early filters are
nudged to behave less alike.

During separation pretraining, a batch of `B` images is passed through each
selected convolution layer. For every image, `pairs_per_layer * 2` random
spatial patches are sampled. A patch has the height and width of the selected
layer's convolution kernel, so applying the filters to one patch produces a
vector of filter responses.

The two representations in each pair are encouraged to point in different
directions, so their cosine similarity should be small.

The separation loss is:

```text
cosine_similarity(response_a, response_b)^2
```

Squaring the cosine similarity asks the representations to be different without
requiring one to be the negative of the other. The loss is averaged over random
patch pairs and over the selected layers.

The config controls how much pretraining to use:

- `baseline`: no separation pretraining
- `layer1`: first convolution only
- `layer12`: first and second convolutions
- `layer123`: first, second, and third convolutions
- `layer1234`: all four convolutions

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
