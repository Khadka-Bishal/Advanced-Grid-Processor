# AdvancedGridProcessor

## Overview

AdvancedGridProcessor is a Python library designed for processing geospatial polygons and unstructured mesh. It provides tools to remove excess nodes and split polygons with more than a specified number of nodes. The toolkit utilizes GeoPandas, Pandas, NumPy, Xugrid, Shapely, and Matplotlib.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Classes](#classes)
  - [GridNodeRemover](#gridnoderemover)
  - [PolygonSplitter](#polygonsplitter)

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
