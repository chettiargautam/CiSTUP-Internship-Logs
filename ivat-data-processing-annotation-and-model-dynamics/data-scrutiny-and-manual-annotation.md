---
description: >-
  Manual annotation of the IVAT data boundaries for helping with clustering
  algorithms
---

# Data Scrutiny and Manual Annotation

```python
import numpy as np
import pandas as pd
```

```python
import plotly.express as ex
```

For clustering boundary, the ground truth is essential to understand the number of clusters expected in the annotation problem.

<figure><img src="../.gitbook/assets/image (8) (1).png" alt=""><figcaption><p>This IVAT Image has a ground truth of 7 clusters</p></figcaption></figure>

The goal is to find the boundaries of each cluster in the form of \[start, end] to understand the exact cluster definition

For such images, the manual annotation has been carried out from my end, and a sample of the end result looks like this

<figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

This annotation has been carried out on 100 IVAT images with simple clusters and 100 IVAT images with complex subclusters

### Cheers!















