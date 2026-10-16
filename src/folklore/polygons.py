"""
Clump masks as polygons (GeoDataFrames / GeoJSON).

Polygons can be written with ``gdf.to_file("clumps.geojson",
driver="GeoJSON")`` and opened in DS9-like tools, QGIS or reused in other
pipelines.
"""

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage import measure

from .io import get_celestial_wcs

__all__ = [
    "mask_to_polygon",
    "build_clump_geodataframe",
    "build_clump_geodataframe_sky",
]


def mask_to_polygon(mask):
    """
    Convert a boolean mask into a shapely polygon using contours.

    The contour is traced at the 0.5 level, i.e. halfway between mask and
    non-mask pixel centres. Coordinates are 0-based pixel coordinates
    ``(x, y)``.

    Parameters
    ----------
    mask : ndarray of bool
        2D mask.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
        Union of all contour polygons; a MultiPolygon if the mask has
        disconnected parts, None if the mask is empty.
    """

    contours = measure.find_contours(mask.astype(float), 0.5)

    polygons = []

    for contour in contours:

        coords = [(c[1], c[0]) for c in contour]  # (x, y)

        if len(coords) < 3:
            continue

        poly = Polygon(coords)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_valid and poly.area > 0:
            polygons.append(poly)

    if not polygons:
        return None

    return unary_union(polygons)


def build_clump_geodataframe(
    union_masks,
    df_union_clumps):
    """
    Build a GeoDataFrame of clump polygons in pixel coordinates.

    Parameters
    ----------
    union_masks : dict
        Dictionary ``{clump_id: mask_union}``.
    df_union_clumps : pandas.DataFrame
        Catalog of clump properties (must contain ``clump_id``). All its
        columns are copied to the GeoDataFrame.

    Returns
    -------
    gdf_clumps : geopandas.GeoDataFrame
        Polygons in pixel coordinates (no CRS).
    """

    rows = []

    clump_lookup = df_union_clumps.set_index("clump_id")

    for cid, mask_union in union_masks.items():

        poly = mask_to_polygon(mask_union)

        if poly is None:
            continue

        record = clump_lookup.loc[cid].to_dict()
        record["clump_id"] = cid
        record["geometry"] = poly

        rows.append(record)

    gdf_clumps = gpd.GeoDataFrame(rows, geometry="geometry")

    gdf_clumps.crs = None

    return gdf_clumps


def build_clump_geodataframe_sky(
    union_masks,
    df_union_clumps,
    wcs):
    """
    Build a GeoDataFrame of clump polygons in sky coordinates.

    The clump contours are first extracted in pixel coordinates and then
    converted vertex-by-vertex into celestial coordinates (RA, Dec) using
    the supplied WCS. Clumps made of several disconnected parts become
    MultiPolygons.

    Parameters
    ----------
    union_masks : dict
        Dictionary ``{clump_id: mask_union}``.
    df_union_clumps : pandas.DataFrame
        Catalog of clump properties (must contain ``clump_id``).
    wcs : astropy.wcs.WCS
        WCS of the detection image or cube (a 3D cube WCS is reduced to
        its celestial part).

    Returns
    -------
    gdf_clumps : geopandas.GeoDataFrame
        Polygons with (RA, Dec) vertices in degrees, CRS ``EPSG:4326``.
    """

    wcs = get_celestial_wcs(wcs)

    def pix_to_sky(xy):
        ra, dec = wcs.pixel_to_world_values(xy[:, 0], xy[:, 1])
        return np.column_stack([ra, dec])

    rows = []

    clump_lookup = df_union_clumps.set_index("clump_id")

    for cid, mask_union in union_masks.items():

        poly_pix = mask_to_polygon(mask_union)

        if poly_pix is None:
            continue

        poly_sky = shapely.transform(poly_pix, pix_to_sky)

        row = clump_lookup.loc[cid].to_dict()
        row["clump_id"] = cid
        row["geometry"] = poly_sky

        rows.append(row)

    gdf_clumps = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326"
    )

    return gdf_clumps
