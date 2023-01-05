---
description: Some more explanation which can be useful for helper functions
---

# Additional Exploration

This is only complementary to the original approach. A few plotting techniques and other considerations have been made here

#### Dataset: Geolife Trajectories

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

```python
mobilityCoordinates = sample_df[[0,1]]
```

<figure><img src="../.gitbook/assets/image (3) (1).png" alt=""><figcaption></figcaption></figure>

```python
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
```

```python
route = sample_df.rename(columns={0:'Latitude', 1:'Longitude'})
```

```python
geometry = [Point(xy) for xy in zip(route['Longitude'], route['Latitude'])]
gdf = GeoDataFrame(route, geometry=geometry)   
```

```python
route.head()
```

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

```python
import folium
```

```python
focusMap = folium.Map(
    location = [np.mean(route.Latitude.values), np.mean(route.Longitude.values)],
    zoom_start = 14)

# Start of the map
folium.Marker([route.Latitude.values[0], route.Longitude.values[0]], 
			popup = 'Start').add_to(focusMap) 

# End of the map
folium.Marker([route.Latitude.values[-1], route.Longitude.values[-1]], 
			popup = 'End').add_to(focusMap)
```

```python
# Obtain the path of movements
motion = mobilityCoordinates.values
```

```python
Motion = []

for coordinate in motion:
    Motion.append((coordinate[0], coordinate[1]))
```

Motion contains the trajectory/sequential data of transit of the vehicles in focus

```python
folium.PolyLine(locations = Motion, 
				line_opacity = 1).add_to(focusMap)
```

<figure><img src="../.gitbook/assets/image (5) (1).png" alt=""><figcaption></figcaption></figure>

```python
# Dataset structure

# Data
#     - Taxi 1
#         - Route 1
#         - Route 2
#         ...
#         - Route n
#     - Taxi 2
#     ...
#     - Taxi n
```

## Naive Computation of the Coverage Function

Naively, we can compute the coverage function as the ratio of the number of unique geohashes traversed by the cabs via transit route data to the number of geohashes within the city.

This also takes in geohashes which are not containing roads, however since only the latitude and longitude are required for calculation, it's computationally faster to obtain.

The entire transit data is shown in the figure below

<figure><img src="../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

Linspacing the bounding boxes into a set number of partitions on the basis of precision coresponding widths will provide the geohash information

```python
# GeoHash length

# 1 5,009.4km x 4,992.6km

# 2 1,252.3km x 624.1km

# 3 156.5km x 156km

# 4 39.1km x 19.5km

# 5 4.9km x 4.9km

# 6 1.2km x 609.4m

# 7 152.9m x 152.4m

# 8 38.2m x 19m

# 9 4.8m x 4.8m

# 10 1.2m x 59.5cm

# 11 14.9cm x 14.9cm
```

```python
extrema = [
    [40.195659, 116.111577],
    [39.747322, 116.778839]
]
```

```python
bottomLeft = (extrema[0][0], extrema[1][1])
bottomRight = (extrema[1][0], extrema[1][1])
topLeft = (extrema[0][0], extrema[0][1])
topRight = (extrema[1][0], extrema[0][1])
```

For precision 7, we have the following split

```python
# For precision 9
cols = np.linspace(bottomLeft[0], bottomRight[0], num=377)
rows = np.linspace(bottomLeft[1], topLeft[1], num=332)
```

We repeat the process of finding geohashes for the entire city and then the transit data and follow with intersection division

```python
allGeohashesInTheGrid = []
```

```python
for row in tqdm(grid):
    for point in row:
        allGeohashesInTheGrid.append(
            geohash.encode(point[1], point[0], precision=7)
            )
```

```python
allGeohashesTraversedByCab = []
```

```python
for trips in Trips:
    for coordinate in trips:
        allGeohashesTraversedByCab.append(
            geohash.encode(coordinate[0], coordinate[1], 7)
        )
```

Uniqueness principle for removing redundancy

```python
allGeohashesInTheGrid = np.array(allGeohashesInTheGrid)
allGeohashesTraversedByCab = np.array(allGeohashesTraversedByCab)
```

```python
allGeohashesInTheGrid = np.unique(allGeohashesInTheGrid)
allGeohashesTraversedByCab = np.unique(allGeohashesTraversedByCab)
```

```python
intersect = 0
total = 0

for hashIndex in tqdm(allGeohashesInTheGrid):
    if hashIndex in allGeohashesTraversedByCab:
        intersect += 1
        total += 1
    else:
        total += 1
```

The coverage is about 2.3%
