# AdvancedGridProcessor

## Overview

AdvancedGridProcessor is a Python library designed for processing geospatial polygons and unstructured mesh. It provides tools to remove excess nodes and split polygons with more than a specified number of nodes. The toolkit utilizes GeoPandas, Pandas, NumPy, Xugrid, Shapely, and Matplotlib.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Classes](#classes)
  - [GridNodeRemover](#gridnoderemover)
  - [PolygonSplitter](#polygonsplitter)
- [Examples](#examples)

## Overview

AdvancedGridProcessor is a Python library designed for processing geospatial polygons and unstructured mesh. It provides tools to remove excess nodes and split polygons with more than a specified number of nodes. The toolkit utilizes GeoPandas, Pandas, NumPy, Xugrid, Shapely, and Matplotlib.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Khadka-Bishal/Advanced-Grid-Processor.git
cd Advanced-Grid-Processor
pip install -r requirements.txt

```

## Classes

### GridNodeRemover

The `GridNodeRemover` class removes excess nodes from polygons/grid faces.

- **Initialization**: `GridNodeRemover(shapefile_path, epsg)`
- **Methods**:
  - `load_and_convert_shapefile()`: Loads and converts the shapefile to the specified EPSG code.
  - `nodes_to_keep_and_remove()`: Identifies nodes to keep and remove.
  - `plot_keep_remove(keept, remt)`: Plots the nodes to keep and remove.
  - `remove_extra_nodes(keept)`: Removes extra nodes from the GeoDataFrame.
  - `save_fixed_grid(newgeo)`: Saves the fixed grid to a NetCDF file.
  - `plot_fixed_grid(xugrid)`: Plots the fixed grid.

### PolygonSplitter

The `PolygonSplitter` class splits polygons/grid faces with more than a specified number of nodes.

- **Initialization**: `PolygonSplitter(shapefile_path, epsg, num_nodes)`
- **Methods**:
  - `load_and_convert_shapefile()`: Loads and converts the shapefile to the specified EPSG code.
  - `find_and_plot_polygons()`: Finds and plots polygons with more than the specified number of nodes.
  - `prepare_split_data()`: Prepares the data for splitting by fixing polygons.
  - `perform_splitting(geo1)`: Splits polygons with excess nodes.
  - `plot_polygon_splits(xugrid1, xugrid_final)`: Plots the split polygons.
  - `save_split_grid(xugrid_final, filename)`: Saves the split grid to a NetCDF file.
 
## Examples

### Example 1: Excess Polygons with Multiple Nodes
![2](https://github.com/user-attachments/assets/4ad6b395-9280-42fe-895c-f5c0361f5729)



### Example 2: Polygons Fixed Using Code
![1](https://github.com/user-attachments/assets/30d127d8-af21-49de-9a71-b26fc46a4204)


### Example 3: Polygons with More Than 6 Nodes Split

![3](https://github.com/user-attachments/assets/611bc75f-f3c4-4d4e-9cb6-e664473efa1b)


