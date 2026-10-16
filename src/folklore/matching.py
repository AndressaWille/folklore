"""
Cross-filter matching of clumps.

A clump detected in one filter is linked to a clump in another filter when
their pixel sets overlap by at least ``min_overlap``. The links form a
graph whose connected components are the physical clumps. Only components
present in at least ``min_filters`` filters are kept, and the final mask of
each physical clump is the union of its pixels over all filters.
"""

import itertools

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "build_clump_pixel_sets",
    "overlap_fraction",
    "compute_clump_overlaps",
    "build_overlap_graph",
    "find_final_clumps",
    "build_final_clump_dataframes",
    "build_union_masks",
]

OVERLAP_COLUMNS = ["filter_1", "label_1", "filter_2", "label_2", "overlap_frac"]
COMPONENT_COLUMNS = ["clump_id", "filter", "label"]


def build_clump_pixel_sets(df_pixels_all):
    """
    Convert a pixel table into pixel sets per clump.

    Parameters
    ----------
    df_pixels_all : pandas.DataFrame
        Table with columns ``filter, label, x, y`` (for example
        ``pixels_all_filters`` from
        :func:`folklore.detection.build_clump_dataframes`).

    Returns
    -------
    clump_pixels : dict
        ``{(filter, label): set of (x, y)}``.
    """

    clump_pixels = {}

    for (filt, lab), g in df_pixels_all.groupby(["filter", "label"]):
        clump_pixels[(filt, lab)] = set(zip(g["x"], g["y"]))

    return clump_pixels


def overlap_fraction(pix_a, pix_b, mode="min"):
    """
    Compute the overlap fraction between two pixel sets.

    Parameters
    ----------
    pix_a, pix_b : set
        Sets of (x, y) pixels.
    mode : {"min", "union"}, optional
        Definition of the overlap fraction:

        - "min": intersection / size of the smaller set (default)
        - "union": intersection / union (Jaccard index)

    Returns
    -------
    float
        Overlap fraction between 0 and 1.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the allowed options.
    """

    inter = len(pix_a & pix_b)

    if inter == 0:
        return 0.0

    if mode == "min":
        return inter / min(len(pix_a), len(pix_b))
    elif mode == "union":
        return inter / len(pix_a | pix_b)
    else:
        raise ValueError("mode must be 'min' or 'union'")


def compute_clump_overlaps(clump_pixels, min_overlap=0.2, mode="min"):
    """
    Compute the overlap between clumps detected in different filters.

    Parameters
    ----------
    clump_pixels : dict
        Output of :func:`build_clump_pixel_sets`.
    min_overlap : float, optional
        Minimum overlap fraction for a pair to be kept.
    mode : {"min", "union"}, optional
        Passed to :func:`overlap_fraction`.

    Returns
    -------
    df_overlap : pandas.DataFrame
        One row per overlapping pair, with columns
        ``filter_1, label_1, filter_2, label_2, overlap_frac``, sorted by
        decreasing overlap.
    """

    overlap_rows = []

    filters = sorted(set(f for f, _ in clump_pixels.keys()))

    for f1, f2 in itertools.combinations(filters, 2):

        clumps_f1 = [(f, l) for (f, l) in clump_pixels if f == f1]
        clumps_f2 = [(f, l) for (f, l) in clump_pixels if f == f2]

        for c1 in clumps_f1:
            for c2 in clumps_f2:

                frac = overlap_fraction(
                    clump_pixels[c1],
                    clump_pixels[c2],
                    mode=mode)

                if frac >= min_overlap:

                    overlap_rows.append({
                        "filter_1": c1[0],
                        "label_1": c1[1],
                        "filter_2": c2[0],
                        "label_2": c2[1],
                        "overlap_frac": frac})

    df_overlap = pd.DataFrame(overlap_rows, columns=OVERLAP_COLUMNS)

    if not df_overlap.empty:
        df_overlap = df_overlap.sort_values(
            "overlap_frac",
            ascending=False)

    return df_overlap


def build_overlap_graph(df_overlap):
    """
    Build a graph connecting clumps with pixel overlap.

    Nodes are ``(filter, label)`` tuples and edges are weighted by the
    overlap fraction.

    Parameters
    ----------
    df_overlap : pandas.DataFrame
        Output of :func:`compute_clump_overlaps`.

    Returns
    -------
    G : networkx.Graph
        Overlap graph. Clumps without any overlap are not included.
    """

    G = nx.Graph()

    for _, row in df_overlap.iterrows():

        n1 = (row["filter_1"], row["label_1"])
        n2 = (row["filter_2"], row["label_2"])

        G.add_edge(
            n1,
            n2,
            weight=row["overlap_frac"]
        )

    return G


def find_final_clumps(G, min_filters=3):
    """
    Identify physical clumps as connected components of the overlap graph.

    Parameters
    ----------
    G : networkx.Graph
        Output of :func:`build_overlap_graph`.
    min_filters : int, optional
        Minimum number of distinct filters in which a clump must be
        detected. Default is 3. Values below 2 have no effect: clumps
        detected in a single filter have no overlaps and are not in the
        graph.

    Returns
    -------
    df_components : pandas.DataFrame
        Columns ``clump_id, filter, label``: the per-filter detections that
        make up each physical clump.
    """

    components = list(nx.connected_components(G))

    valid_components = []

    for comp in components:

        filters_in_comp = {f for (f, l) in comp}

        if len(filters_in_comp) >= min_filters:
            valid_components.append(comp)

    rows = []

    for cid, comp in enumerate(valid_components):

        for filt, lab in comp:

            rows.append({
                "clump_id": cid,
                "filter": filt,
                "label": lab
            })

    df_components = pd.DataFrame(rows, columns=COMPONENT_COLUMNS)

    return df_components


def build_final_clump_dataframes(
    clumps_by_filter,
    pixels_by_filter,
    df_components):
    """
    Build the final clump and pixel tables.

    Keeps only the per-filter detections that belong to a physical clump
    and attaches its ``clump_id``.

    Parameters
    ----------
    clumps_by_filter, pixels_by_filter : dict
        Outputs of :func:`folklore.detection.build_clump_dataframes`.
    df_components : pandas.DataFrame
        Output of :func:`find_final_clumps`.

    Returns
    -------
    df_clumps_final : pandas.DataFrame
        Per-filter clump table with a ``clump_id`` column.
    df_pixels_final : pandas.DataFrame
        Per-filter pixel table with a ``clump_id`` column.
    """

    df_clumps_all = pd.concat(
        clumps_by_filter.values(),
        ignore_index=True)

    df_pixels_all = pd.concat(
        pixels_by_filter.values(),
        ignore_index=True)

    df_clumps_final = df_clumps_all.merge(
        df_components,
        on=["filter", "label"],
        how="inner")

    df_pixels_final = df_pixels_all.merge(
        df_components,
        on=["filter", "label"],
        how="inner")

    return df_clumps_final, df_pixels_final


def build_union_masks(df_pixels_final, image_shape):
    """
    Build union pixel masks for each final clump.

    Parameters
    ----------
    df_pixels_final : pandas.DataFrame
        Table containing ``x``, ``y`` and ``clump_id`` for all pixels.
    image_shape : tuple
        Shape of the image (ny, nx).

    Returns
    -------
    union_masks : dict
        Dictionary ``{clump_id: boolean mask}``.
    """

    ny_pix, nx_pix = image_shape
    union_masks = {}

    for cid, g in df_pixels_final.groupby("clump_id"):

        mask_union = np.zeros((ny_pix, nx_pix), dtype=bool)

        mask_union[g["y"].values.astype(int), g["x"].values.astype(int)] = True

        union_masks[cid] = mask_union

    return union_masks
