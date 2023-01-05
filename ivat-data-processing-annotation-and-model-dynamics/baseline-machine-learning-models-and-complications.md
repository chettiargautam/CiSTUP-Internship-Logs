---
description: The results and benchmarks while training using Tensorflow CNN Models
---

# Baseline Machine Learning models and complications

## Considerations

`Input: (None, 500, 500)`\
`Output: (None, 500, 1)`

On training a model with even 10M parameters and using only 2D-CNN architectures and minimal fully connected layers, for a default batch size of 32, it requires almost 24GB of VRAM for training.

This eliminates the scope of using deeper nets such as VGG19 or InceptionV3 due to computational considerations.
