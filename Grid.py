# Import necessary libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import xugrid as xu
import shapely
import matplotlib.pyplot as plt

class GridNodeRemover:
    """
    Class to process polygons by removing excess nodes.
    """

    def __init__(self, shapefile_path, epsg):
        """
        Initialize the GridNodeRemover with a shapefile and EPSG code.

        Parameters:
        shapefile_path (str): Path to the shapefile.
        epsg (int): EPSG code for coordinate reference system.
        """
        self.shapefile_path = shapefile_path
        self.epsg = epsg
        self.geo, self.xugrid = self.load_and_convert_shapefile()

    def load_and_convert_shapefile(self):
        """
        Load and convert the shapefile to the specified EPSG code.

        Returns:
        tuple: GeoDataFrame and Xugrid dataset.
        """
        geo = gpd.read_file(self.shapefile_path)
        geo = geo.to_crs(epsg=self.epsg)  # Change to UTM for meters
        xugrid = xu.UgridDataset.from_geodataframe(geo)
        return geo, xugrid

    def nodes_to_keep_and_remove(self):
        """
        Identify nodes to keep and remove based on connectivity.

        Returns:
        tuple: Arrays of node indices to keep and remove.
        """
        fnc = self.xugrid.ugrid.grid.face_node_connectivity
        fncf = fnc.flatten()
        k = fncf == -1
        fncf = np.delete(fncf, k)
        bc = np.bincount(fncf)
        keep = np.where(bc > 2)[0]
        rem = np.where(bc < 3)[0]

        enc = self.xugrid.ugrid.grid.edge_node_connectivity
        efc = self.xugrid.ugrid.grid.edge_face_connectivity

        exte = np.where((efc[:, 1] == -1))[0]
        keepe = np.unique(enc[exte].flatten())

        keept = np.union1d(keep, keepe)
        remt = np.setdiff1d(rem, keept)

        return keept, remt

    def plot_keep_remove(self, keept, remt):
        """
        Plot the nodes to keep and remove on the grid.

        Parameters:
        keept (array): Node indices to keep.
        remt (array): Node indices to remove.
        """
        self.xugrid.ugrid.grid.plot(figsize=(10, 10))
        nodes = self.xugrid.ugrid.grid.node_coordinates

        plt.plot(nodes[keept, 0], nodes[keept, 1], '*m')
        plt.plot(nodes[remt, 0], nodes[remt, 1], '.k', alpha=.8)
        plt.show()

    def fix_poly(self, r, fnc, keept, fid):
        """
        Fix a polygon by removing nodes and keeping necessary ones.

        Parameters:
        r (GeoSeries): Row of the GeoDataFrame.
        fnc (array): Face node connectivity array.
        keept (array): Node indices to keep.
        fid (array): Feature IDs.

        Returns:
        Polygon: Updated polygon geometry.
        """
        findx = np.where((fid == r['FID']))[0][0]
        _, fni, _ = np.intersect1d(fnc[findx], keept, return_indices=True)
        fni.sort()
        p = np.array(r.geometry.exterior.coords)[fni]
        if p.shape[0] < 3:
            return r.geometry
        else:
            return shapely.Polygon(np.array(r.geometry.exterior.coords)[fni])

    def remove_extra_nodes(self, keept):
        """
        Remove extra nodes from the GeoDataFrame.

        Parameters:
        keept (array): Node indices to keep.

        Returns:
        GeoDataFrame: Updated GeoDataFrame with extra nodes removed.
        """
        fid = self.xugrid['FID'].to_numpy()
        newgeo = self.geo.copy(deep=True)
        newgeo['oldgeom'] = newgeo['geometry']
        newgeo['geometry'] = newgeo.apply(self.fix_poly, fnc=self.xugrid.ugrid.grid.face_node_connectivity, keept=keept, fid=fid, axis=1)
        newgeo = newgeo.drop('oldgeom', axis=1)
        return newgeo

    def save_fixed_grid(self, newgeo):
        """
        Save the fixed grid to a NetCDF file.

        Parameters:
        newgeo (GeoDataFrame): Updated GeoDataFrame.

        Returns:
        UgridDataset: Xugrid dataset.
        """
        xugrid = xu.UgridDataset.from_geodataframe(newgeo)
        xugrid.ugrid.to_netcdf("SmallerAreaMultiPointsFixed_net.nc")
        return xugrid

    def plot_fixed_grid(self, xugrid):
        """
        Plot the fixed grid.

        Parameters:
        xugrid (UgridDataset): Xugrid dataset.
        """
        xugrid.ugrid.grid.plot(figsize=(10, 10))
        nodes = xugrid.ugrid.grid.node_coordinates
        plt.plot(nodes[:, 0], nodes[:, 1], '.k', ms=5)
        plt.show()

class PolygonSplitter:
    """
    Class to split polygons with more than a specified number of nodes.
    """

    def __init__(self, shapefile_path, epsg, num_nodes=6):
        """
        Initialize the PolygonSplitter with a shapefile, EPSG code, and maximum number of nodes.

        Parameters:
        shapefile_path (str): Path to the shapefile.
        epsg (int): EPSG code for coordinate reference system.
        num_nodes (int): Maximum number of nodes in a polygon.
        """
        self.shapefile_path = shapefile_path
        self.epsg = epsg
        self.num_nodes = num_nodes
        self.geo, self.xugrid = self.load_and_convert_shapefile()

    def load_and_convert_shapefile(self):
        """
        Load and convert the shapefile to the specified EPSG code.

        Returns:
        tuple: GeoDataFrame and Xugrid dataset.
        """
        geo = gpd.read_file(self.shapefile_path)
        geo = geo.to_crs(epsg=self.epsg)  # Change to UTM for meters
        xugrid = xu.UgridDataset.from_geodataframe(geo)
        return geo, xugrid

    def find_and_plot_polygons(self):
        """
        Find and plot polygons with more than the specified number of nodes.
        """
        fnc = self.xugrid.ugrid.grid.face_node_connectivity
        fnn = np.sum(fnc > -1, axis=1)  # Number of nodes per face
        fp = np.where((fnn > self.num_nodes))[0]  # Array of face indices with more than num_nodes nodes
        print(f'Indices of polygons with more than {self.num_nodes} nodes \n {fp} \n')
        self.fidp = self.xugrid.FID.to_numpy()[fp]  # Problem faces
        self.xugrid.ugrid.grid.plot(figsize=(15, 25))
        faces = self.xugrid.ugrid.grid.face_coordinates[fp]
        plt.plot(faces[:, 0], faces[:, 1], '*m')
        plt.gca().set_aspect('equal')
        plt.show()

    def fix_poly(self, r, fnc, keept, fid):
        """
        Fix a polygon by removing nodes and keeping necessary ones.

        Parameters:
        r (GeoSeries): Row of the GeoDataFrame.
        fnc (array): Face node connectivity array.
        keept (array): Node indices to keep.
        fid (array): Feature IDs.

        Returns:
        Polygon: Updated polygon geometry.
        """
        findx = np.where((fid == r['FID']))[0][0]
        _, fni, _ = np.intersect1d(fnc[findx], keept, return_indices=True)
        fni.sort()
        p = np.array(r.geometry.exterior.coords)[fni]
        if p.shape[0] < 3:
            return r.geometry
        else:
            return shapely.Polygon(np.array(r.geometry.exterior.coords)[fni])

    def prepare_split_data(self):
        """
        Prepare the data for splitting by fixing polygons with excess nodes.

        Returns:
        GeoDataFrame: Updated GeoDataFrame with fixed polygons.
        """
        fnc = self.xugrid.ugrid.grid.face_node_connectivity
        fncf = fnc.flatten()
        k = fncf == -1
        fncf = np.delete(fncf, k)
        bc = np.bincount(fncf)
        keep = np.where(bc > 2)[0]
        rem = np.where(bc < 3)[0]

        enc = self.xugrid.ugrid.grid.edge_node_connectivity
        efc = self.xugrid.ugrid.grid.edge_face_connectivity
        exte = np.where((efc[:, 1] == -1))[0]
        keepe = np.unique(enc[exte].flatten())
        keept = np.union1d(keep, keepe)
        remt = np.setdiff1d(rem, keept)

        fid = self.xugrid['FID'].to_numpy()
        geo1 = self.geo.copy(deep=True

)
        geo1['oldgeom'] = geo1['geometry']
        geo1['geometry'] = geo1.apply(self.fix_poly, fnc=fnc, keept=keept, fid=fid, axis=1)
        geo1 = geo1.drop('oldgeom', axis=1)
        geo1 = geo1.drop(['CellIndex', 'MeshName'], axis=1)
        geo1.index.name = 'mesh2d_nFaces'
        return geo1

    def split_poly(self, r, fnc, fid):
        """
        Split a polygon into two smaller polygons.

        Parameters:
        r (GeoSeries): Row of the GeoDataFrame.
        fnc (array): Face node connectivity array.
        fid (array): Feature IDs.

        Returns:
        tuple: Two new polygon geometries.
        """
        findx = np.where((fid == r['FID']))[0][0]
        n = fnc[findx]
        ni = np.arange(0, len(n[n > -1]))
        p0 = np.array(r.geometry.exterior.coords)[:int(np.ceil(len(ni) / 2)) + 1]
        pn = np.array(r.geometry.exterior.coords)[int(np.ceil(len(ni) / 2)):]
        if p0.shape[0] < 3 or pn.shape[0] < 3:
            print('problem', p0, findx, 'fid is ', r['FID'])
            print('problem', pn)
            return r.geometry
        else:
            return shapely.Polygon(p0), shapely.Polygon(pn)

    def perform_splitting(self, geo1):
        """
        Apply the split_poly method to the GeoDataFrame to split polygons with excess nodes.

        Parameters:
        geo1 (GeoDataFrame): GeoDataFrame with fixed polygons.

        Returns:
        UgridDataset: Xugrid dataset with split polygons.
        """
        xugrid1 = xu.UgridDataset.from_geodataframe(geo1)
        fnc = xugrid1.ugrid.grid.face_node_connectivity
        fnn = np.sum(fnc > -1, axis=1)  # Number of nodes per face
        fp = np.where((fnn > self.num_nodes))[0]  # Array of face indices with more than num_nodes nodes
        fidp = xugrid1.FID.to_numpy()[fp]  # Problem faces

        geo2 = geo1.copy(deep=True)
        nwp = geo2.loc[fidp].copy()

        ab = geo2.loc[fidp].apply(self.split_poly, fnc=fnc, fid=fidp, axis=1, result_type='expand')
        nwp['geometry'] = ab.iloc[:, 1]
        geo2.loc[fidp, 'geometry'] = ab.iloc[:, 0]
        nwp = nwp.set_index(np.arange(geo2['FID'].max() + 1, len(nwp) + geo2['FID'].max() + 1))
        nwp['OldFID'] = nwp['FID']
        nwp['FID'] = nwp.index
        geo2['OldFID'] = geo2['FID']
        geo2['SplitPoly'] = 0
        geo2['PairedPolyInd'] = np.nan
        geo2.loc[fidp, 'SplitPoly'] = 1
        geo2.loc[fidp, 'PairedPolyInd'] = nwp['FID']
        nwp['SplitPoly'] = 1
        nwp['PairedPolyInd'] = nwp['OldFID']
        geo2 = pd.concat([geo2, nwp])
        geo2['Count'] = geo2['geometry'].apply(lambda x: len(x.exterior.coords) - 1)
        geo2['Area'] = geo2['geometry'].area
        geo2['Length'] = geo2['geometry'].length

        return xu.UgridDataset.from_geodataframe(geo2)

    def plot_polygon_splits(self, xugrid1, xugrid_final):
        """
        Plot the split polygons.

        Parameters:
        xugrid1 (UgridDataset): Initial Xugrid dataset.
        xugrid_final (UgridDataset): Final Xugrid dataset with split polygons.
        """
        xugrid1.ugrid.grid.plot(figsize=(8, 8), color='k')
        xugrid_final.ugrid.grid.plot(linestyle='--', alpha=.5)

        faces = xugrid_final.ugrid.grid.face_coordinates
        plt.plot(faces[:, 0], faces[:, 1], '*m')

        ppi = 500
        pp = xugrid_final.ugrid.grid.to_dataframe().loc[ppi, 'geometry']
        plt.axis((pp.buffer(1e3).bounds[0],
                  pp.buffer(1e3).bounds[2],
                  pp.buffer(1e3).bounds[1],
                  pp.buffer(1e3).bounds[3],))
        plt.gca().set_aspect('equal')
        plt.show()

    def save_split_grid(self, xugrid_final, filename):
        """
        Save the split grid to a NetCDF file.

        Parameters:
        xugrid_final (UgridDataset): Final Xugrid dataset with split polygons.
        filename (str): Name of the file to save.
        """
        xugrid_final.ugrid.to_netcdf(filename)

# Example usage for Section 1
shapefile_path = "Test2DFlowAreaMesh_Polygon.shp"
processor = PolygonProcessor(shapefile_path, epsg=32618)

keept, remt = processor.nodes_to_keep_and_remove()
processor.plot_keep_remove(keept, remt)

newgeo = processor.remove_extra_nodes(keept)
xugrid = processor.save_fixed_grid(newgeo)
processor.plot_fixed_grid(xugrid)

# Example usage for Section 2
splitter = PolygonSplitter(shapefile_path, epsg=32618, num_nodes=6)

splitter.find_and_plot_polygons()

geo1 = splitter.prepare_split_data()
xugrid1 = xu.UgridDataset.from_geodataframe(geo1)

xugrid_final = splitter.perform_splitting(geo1)
splitter.plot_polygon_splits(xugrid1, xugrid_final)

splitter.save_split_grid(xugrid_final, "test_split_pol_net.nc")
