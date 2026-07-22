"""
Modification clause Sankey visualisation.

Given:
    metadata   : pd.DataFrame with columns:
                   - 'year'                : publication date as YYYYMMDD (int or str)
                   - 'p'                  : platform name
                   - 's'                  : sentence text
                   - 'modification_recoded': 1 if the sentence is a modification
                                            clause, 0 otherwise
    embeddings : np.ndarray, shape (N, D) — row i is the embedding for
                 metadata.iloc[i]

Produces an interactive Plotly Sankey diagram where:
    - Each node   = one (platform, publication_date) pair that contains at
                    least one modification clause
    - Each edge   = a directed link from an earlier publication to a later one
                    (same or different platform) when their modification-clause
                    similarity exceeds a threshold
    - Edge value  = the similarity score (controls band width in the Sankey)

Similarity methods
------------------
'mean'     : cosine similarity between the mean (centroid) embeddings of the
             two modification-clause sets.  Fast; robust when sets are large.
'max'      : maximum pairwise cosine similarity across all sentence pairs.
             Fires if *any* sentence pair is highly similar — good for
             detecting partial overlap.
'mean_pairwise' : mean of the full pairwise cosine similarity matrix.
             Richer than centroid similarity; captures distributional overlap.
'median_pairwise' : median of the pairwise matrix; robust to outlier pairs.

Thresholding strategy
---------------------
'absolute' : keep edges where similarity >= threshold  (default)
'topk'     : for each source node keep only the top-k most similar targets
'percentile': keep edges above the p-th percentile of all computed similarities
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mod_index_lookup(metadata: pd.DataFrame) -> dict[tuple, list[int]]:
    """
    Return a dict mapping (platform, date) -> list of integer row indices
    for rows where modification_recoded == 1.
    """
    mod = metadata[metadata["modification_recoded"] == 1]
    lookup: dict[tuple, list[int]] = {}
    for (date, platform), grp in mod.groupby(["year", "platform"]):
        lookup[(str(platform), str(date))] = grp.index.tolist()
    return lookup


def _node_label(platform: str, date: str) -> str:
    """Human-readable label: 'Twitter 20180501'."""
    return f"{platform}\n{date}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pairwise similarity between two sets of embeddings
# ─────────────────────────────────────────────────────────────────────────────

SIMILARITY_METHODS = ("mean", "max", "mean_pairwise", "median_pairwise")


def _pair_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    method: str,
) -> float:
    """
    Compute a scalar similarity between two embedding matrices.

    Parameters
    ----------
    emb_a, emb_b : ndarray of shape (M, D) and (K, D).
    method       : one of SIMILARITY_METHODS.
    """
    if method == "mean":
        ca = emb_a.mean(axis=0, keepdims=True)
        cb = emb_b.mean(axis=0, keepdims=True)
        return float(sk_cosine(ca, cb)[0, 0])

    matrix = sk_cosine(emb_a, emb_b)   # (M, K)

    if method == "max":
        return float(matrix.max())
    if method == "mean_pairwise":
        return float(matrix.mean())
    if method == "median_pairwise":
        return float(np.median(matrix))

    raise ValueError(
        f"Unknown method {method!r}. Choose from {SIMILARITY_METHODS}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Edge computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mod_edges(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    method: str = "mean",
    threshold_strategy: str = "absolute",
    threshold: float = 0.70,
    topk: int = 3,
    percentile: float = 75.0,
    future_only: bool = True,
) -> pd.DataFrame:
    """
    Compute directed edges between (platform, date) publication nodes based on
    the similarity of their modification-clause sentence sets.

    Parameters
    ----------
    metadata           : see module docstring.
    embeddings         : ndarray (N, D).
    method             : similarity aggregation — one of SIMILARITY_METHODS.
    threshold_strategy : 'absolute' | 'topk' | 'percentile'
    threshold          : used when threshold_strategy='absolute'.
    topk               : used when threshold_strategy='topk'.
    percentile         : used when threshold_strategy='percentile' (0–100).
    future_only        : if True, only draw edges from earlier to later dates.

    Returns
    -------
    DataFrame with columns:
        src_platform, src_date, tgt_platform, tgt_date, similarity
    """
    if method not in SIMILARITY_METHODS:
        raise ValueError(f"method must be one of {SIMILARITY_METHODS}")
    if threshold_strategy not in ("absolute", "topk", "percentile"):
        raise ValueError("threshold_strategy must be 'absolute', 'topk', or 'percentile'")

    lookup = _mod_index_lookup(metadata)
    nodes  = sorted(lookup.keys())          # list of (platform, date) tuples

    # ------------------------------------------------------------------
    # Compute all pairwise similarities
    # ------------------------------------------------------------------
    records = []
    for i, (plat_a, date_a) in enumerate(nodes):
        for j, (plat_b, date_b) in enumerate(nodes):
            if i == j:
                continue
            if future_only and date_b <= date_a:
                continue

            emb_a = embeddings[lookup[(plat_a, date_a)]]
            emb_b = embeddings[lookup[(plat_b, date_b)]]
            sim   = _pair_similarity(emb_a, emb_b, method)

            records.append({
                "src_platform": plat_a,
                "src_date":     date_a,
                "tgt_platform": plat_b,
                "tgt_date":     date_b,
                "similarity":   sim,
            })

    if not records:
        return pd.DataFrame(columns=["src_platform","src_date",
                                     "tgt_platform","tgt_date","similarity"])

    edges = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Apply threshold strategy
    # ------------------------------------------------------------------
    if threshold_strategy == "absolute":
        edges = edges[edges["similarity"] >= threshold]

    elif threshold_strategy == "topk":
        edges = (
            edges
            .sort_values("similarity", ascending=False)
            .groupby(["src_platform", "src_date"])
            .head(topk)
            .reset_index(drop=True)
        )

    elif threshold_strategy == "percentile":
        cutoff = np.percentile(edges["similarity"].values, percentile)
        edges  = edges[edges["similarity"] >= cutoff]

    return edges.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sankey figure
# ─────────────────────────────────────────────────────────────────────────────

# One colour per platform (cycles if > 10 platforms)
_PALETTE = [
    "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
    "#378ADD", "#EF9F27", "#639922", "#888780",
    "#A855F7", "#EC4899",
]


def build_sankey(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    method: str = "mean",
    threshold_strategy: str = "absolute",
    threshold: float = 0.70,
    topk: int = 3,
    percentile: float = 75.0,
    future_only: bool = True,
    title: str = "Evolution of modification clauses across ToS publications",
) -> go.Figure:
    """
    Build an interactive Plotly Sankey diagram of modification-clause evolution.

    Each node is a (platform, publication_date) pair that contains at least one
    modification clause.  Edges indicate semantic similarity above the chosen
    threshold between consecutive publications.

    Parameters
    ----------
    metadata, embeddings   : see module docstring.
    method                 : similarity method — one of SIMILARITY_METHODS.
    threshold_strategy     : 'absolute' | 'topk' | 'percentile'
    threshold              : absolute cutoff (used when strategy='absolute').
    topk                   : edges per source node (used when strategy='topk').
    percentile             : percentile cutoff (used when strategy='percentile').
    future_only            : only draw forward-in-time edges.
    title                  : figure title.

    Returns
    -------
    go.Figure
    """
    edges_df = compute_mod_edges(
        metadata, embeddings,
        method=method,
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        topk=topk,
        percentile=percentile,
        future_only=future_only,
    )

    if edges_df.empty:
        raise ValueError(
            "No edges survive the threshold — try lowering it or changing the strategy."
        )

    # ------------------------------------------------------------------
    # Build node list (union of src and tgt appearing in surviving edges)
    # ------------------------------------------------------------------
    src_nodes = list(zip(edges_df["src_platform"], edges_df["src_date"]))
    tgt_nodes = list(zip(edges_df["tgt_platform"], edges_df["tgt_date"]))
    all_nodes  = sorted(set(src_nodes + tgt_nodes), key=lambda x: (x[1], x[0]))

    node_index = {n: i for i, n in enumerate(all_nodes)}

    # ------------------------------------------------------------------
    # Colour map: one colour per platform
    # ------------------------------------------------------------------
    platforms  = sorted({n[0] for n in all_nodes})
    plat_color = {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(platforms)}

    node_colors = [plat_color[n[0]] for n in all_nodes]
    node_labels = [_node_label(n[0], n[1]) for n in all_nodes]

    # Count modification sentences per node for hover info
    lookup     = _mod_index_lookup(metadata)
    node_counts = [len(lookup.get(n, [])) for n in all_nodes]

    node_hover = [
        f"<b>{n[0]}</b><br>Date: {n[1]}<br>Mod. clauses: {c}"
        for n, c in zip(all_nodes, node_counts)
    ]

    # ------------------------------------------------------------------
    # Build Sankey link arrays
    # ------------------------------------------------------------------
    sources = [node_index[(r.src_platform, r.src_date)] for _, r in edges_df.iterrows()]
    targets = [node_index[(r.tgt_platform, r.tgt_date)] for _, r in edges_df.iterrows()]
    values  = edges_df["similarity"].tolist()

    # Edge colours inherit from the source node, semi-transparent
    def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [
        _hex_to_rgba(plat_color[edges_df.iloc[i]["src_platform"]])
        for i in range(len(edges_df))
    ]

    link_hover = [
        (
            f"{r.src_platform} ({r.src_date})"
            f" → {r.tgt_platform} ({r.tgt_date})"
            f"<br>similarity ({method}): {r.similarity:.3f}"
        )
        for _, r in edges_df.iterrows()
    ]

    # ------------------------------------------------------------------
    # Assemble figure
    # ------------------------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=18,
            line=dict(color="white", width=0.5),
            label=node_labels,
            color=node_colors,
            customdata=node_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=link_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))

    # Legend annotation (Sankey doesn't support native legend)
    legend_annotations = []
    for i, p in enumerate(platforms):
        legend_annotations.append(dict(
            x=1.01, y=1.0 - i * 0.06,
            xref="paper", yref="paper",
            xanchor="left",
            text=f"<span style='color:{plat_color[p]}'>■</span> {p}",
            showarrow=False,
            font=dict(size=12),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        font=dict(size=11),
        margin=dict(l=20, r=160, t=60, b=20),
        height=max(500, len(all_nodes) * 40 + 120),
        annotations=legend_annotations,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Convenience: return edges DataFrame alongside figure
# ─────────────────────────────────────────────────────────────────────────────

def build_sankey_with_edges(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    **kwargs,
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Same as build_sankey() but also returns the edges DataFrame so you can
    inspect which publications were connected and why.

    Returns
    -------
    (fig, edges_df)
    """
    edges_df = compute_mod_edges(metadata, embeddings, **{
        k: kwargs[k] for k in (
            "method", "threshold_strategy", "threshold",
            "topk", "percentile", "future_only",
        ) if k in kwargs
    })
    fig = build_sankey(metadata, embeddings, **kwargs)
    return fig, edges_df
