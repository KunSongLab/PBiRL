import json
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.ops import unary_union

def voronoi_finite_polygons_2d(vor, radius=500):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

data = gpd.read_file("../scenarios/data/Kowloon/nodes.csv", crs=2326)
with open("../scenarios/data/Kowloon/node_clusters.json") as f:
    node_cluster = json.load(f)

node_cluster_series = {"osmid": [], "region": []}
for key, value in node_cluster.items():
    for each in value:
        node_cluster_series["osmid"].append(each)
        node_cluster_series["region"].append(int(key))
node_cluster_series = pd.DataFrame(node_cluster_series)

boundary = gpd.read_file("../scenarios/data/Kowloon/kowloon.geojson", crs=4326).to_crs(epsg=2326)
data["osmid"] = data["osmid"].astype(np.int64)
data["geometry"] = gpd.GeoSeries.from_wkt(data["wkt"])
data["real_x"] = data.apply(lambda p: p["geometry"].x, axis=1)
data["real_y"] = data.apply(lambda p: p["geometry"].y, axis=1)
data = data.join(node_cluster_series.set_index("osmid"), on="osmid")
data = data[data["region"] != 28]
sns.scatterplot(data=data, x="real_x", y="real_y", hue="region")
plt.show()

points = np.array([data["real_x"].values, data["real_y"].values]).T
vor = Voronoi(points)
new_regions, new_vertices = voronoi_finite_polygons_2d(vor)

shapes = []
print(new_vertices)
for i, each in enumerate(new_regions):
    shapes.append(Polygon(new_vertices[item] for item in each))

data["geometry"] = shapes
data = gpd.clip(data, boundary)
colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff",
          "#00441b", "#006d2c", "#238b45", "#41ab5d", "#74c476", "#a1d99b", "#c7e9c0", "#e5f5e0", "#f7fcf5",
          "#7f2704", "#a63603", "#d94801", "#f16913", "#fd8d3c", "#fdae6b", "#fdd0a2", "#fee6ce", "#fff5eb",
          "#3f007d", "#54278f", "#6a51a3", "#807dba", "#807dba", "#bcbddc", "#dadaeb", "#efedf5", "#fcfbfd",
          ]
data["color"] = data.apply(lambda x: colors[x["region"]], axis=1)
data.plot(edgecolor="none", column="region")
plt.show()

print(data["geometry"])
res = {"geometry": []}
temp_groupby = data.groupby("region")
for key in range(28):
    temp = temp_groupby.get_group(key)
    temp_boundary = gpd.GeoSeries(unary_union(temp["geometry"]))
    # temp_boundary.plot()

    res["geometry"].append(temp_boundary.set_crs(epsg=2326).to_crs(epsg=4326).geometry.values[0])
    print(temp_boundary.set_crs(epsg=2326).to_crs(epsg=4326).geometry.values[0])
res = gpd.GeoDataFrame(res)

res.to_file("region_shape.json", driver="GeoJSON")