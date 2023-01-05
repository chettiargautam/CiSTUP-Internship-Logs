---
description: >-
  Implementing primary analyses on the TDrive Taxi Cab Transit Dataset
  consisting of roughly 10,000 route data. Also the base code for implementing
  the Coverage function has been discussed here.
---

# Geospatial Analysis on the Beijing TDrive Dataset and Coverage Function

```python
import geohash
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
```

### Understanding Mapping in terms of (Latitude, Longitude) and Geohashes

Generally, a spatial point on the earth can be represented in terms of Latitude and Longitude. These are predefined and specific to locations. A geohash is an encoded version of a bounding box ranging between two extrema such coordinate points.

`geohash = geohash.encode(coordinate[1], coordinate[0], precision=precision)`

The precision determines the cross-sectional lengths of the bounding box. A precision of 7 will cover significantly more area than a precision of 9, which is to be used while considering scenarios that adapt to larger data.

### TDrive Dataset

TDrive Dataset contains the taxi transit data for the city of Beijing. It has 10357 rows/entries for the same.

```python
basePath = r"taxi_log_2008_by_id/"
directory = os.fsencode(basePath)
    
for file in tqdm(os.listdir(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"): 
         routes_txt.append(filename)

for route_plt in tqdm(routes_txt):
    try:
        routes.append(np.genfromtxt(basePath + route_plt, delimiter=',', skip_header=0))
    except:
        pass
```

The coverage function for this dataset only requires the set of Latitudes and Longitudes, so they are isolated accordingly

```python
trips = []

for route in tqdm(routes):
    try:
        trips.append(route[:,-2:])
    except IndexError:
        continue
```

A sample transit route looks like this

array(\[\[116.51172, 39.92123], \[116.51135, 39.93883], \[116.51135, 39.93883], ..., \[116.53174, 39.91536], \[116.57156, 39.90263], \[116.54723, 39.90841]])

Following this, the conversion to standard format of \[Longitude, Latitude] is done as follows

```python
Trips = []

for trip in trips:
    Trip = []
    for coordinate in trip:
        Trip.append((coordinate[1], coordinate[0]))
    Trips.append(Trip)
```

## Open Street Maps

We use open street maps python library to obtain the road data in terms of coordinate points, which is essential for the calculation of the coverage function

```python
import osmnx as ox
city = ox.geocode_to_gdf('Beijing, China')
ax = ox.project_gdf(city).plot()
_ = ax.axis('off')
```

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption></figcaption></figure>

```python
ox.plot_graph(ox.graph_from_place('Beijing, China'))
```

<figure><img src="../.gitbook/assets/image (4) (1).png" alt=""><figcaption></figcaption></figure>

```python
G = ox.graph_from_place('Beijing, China')
nodes, edges = ox.graph_to_gdfs(G)
nodes.head()
```

<figure><img src="../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

```python
nodes[['y','x']].values
```

array(\[\[ 39.8344898, 116.4622121], \[ 39.8325259, 116.4567484], \[ 39.8293845, 116.3229152], ..., \[ 40.2533273, 115.9257606], \[ 40.4509362, 115.7842617], \[ 40.4490015, 115.7824968]])

```python
BeijingRoadCoordinates = []

for node in tqdm(nodes[['y','x']].values):
    BeijingRoadCoordinates.append([node[0], node[1]])   
```

BeijingRoadCoorindates contains all the Road Lats and Longs for the entire city. Mapping each of these coordinates to their respective geohashes of a fixed precision is key for finding coverage factor.

```python
BeijingRoadGeohashes = []

for coordinate in tqdm(BeijingRoadCoordinates):
    hash = geohash.encode(coordinate[0], coordinate[1], precision=7)
    BeijingRoadGeohashes.append(hash)
```

Many of the obtained geohashes will be similar due to road coordinates being common to them. Hence it is important to only take the unique values of these geohashes.

```python
BeijingRoadGeohashes = np.array(BeijingRoadGeohashes)
```

```python
BeijingRoadGeohashes = np.unique(BeijingRoadGeohashes)
```

## Finding the coordinates and geohashes traversed by the cabs

```python
allGeohashesTraversedByCab = []

for trips in tqdm(Trips):
    for coordinate in trips:
        try:
            allGeohashesTraversedByCab.append(
                geohash.encode(coordinate[0], coordinate[1], 7)
            )
        except:
            pass
```

```python
allGeohashesTraversedByCab = np.array(allGeohashesTraversedByCab)
```

```python
allGeohashesTraversedByCab = np.unique(allGeohashesTraversedByCab)
```

The same uniqueness principle has been applied to the geohashes of the cabs to avoid redundancy for the coverage function.

$$coverage(geohash_{focus}, geohash_{total}, t) = \frac{Intersection_{unique}(geohash_{focus}, geohash_{total})}{Total_{unique}(geohash_{total})} \: given\:t$$

Python implementation in a one-vs-all structure

```python
intersect = 0
total = 0

for hashIndex in tqdm(BeijingRoadGeohashes):
    if hashIndex in allGeohashesTraversedByCab:
        intersect += 1
        total += 1
    else:
        total += 1
```

```python
"The coverage factor for the TDrive Dataset is :" + str(intersect / total)
```

Which is 45.29%
