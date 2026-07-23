import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from IPython.display import display
from tqdm.auto import tqdm


def _prepare_nodes_and_embeddings(metadata_clause, embeddings_clause):
    nodes = metadata_clause.copy().reset_index(drop=True)
    nodes = nodes.rename(columns={"year_int": "year"})

    required_cols = ["platform", "year", "sentence"]
    for c in required_cols:
        if c not in nodes.columns:
            raise ValueError(f"metadata_clause is missing required column: {c}")

    mask = (
        nodes["platform"].notna()
        & nodes["sentence"].notna()
        & pd.to_numeric(nodes["year"], errors="coerce").notna()
    )

    nodes = nodes.loc[mask].copy().reset_index(drop=True)
    nodes["year"] = pd.to_numeric(nodes["year"], errors="coerce").astype(int)
    nodes["sentence"] = nodes["sentence"].astype(str).str.strip()
    nodes = nodes[nodes["sentence"].str.len() > 0].reset_index(drop=True)

    emb = np.asarray(embeddings_clause)
    emb = emb[mask.to_numpy()]

    # L2-normalize rows to make dot product equal cosine similarity.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    nodes["node_id"] = np.arange(len(nodes))
    return nodes, emb


def _build_time_influence_edges(nodes, emb, sim_threshold=0.88, max_lag=5, k_closest=2):
    """
    Build cross-platform influence edges with strict first-occurrence attribution.

    Important: to preserve first-occurrence semantics, we evaluate all valid prior
    candidates (not just closest-in-time subsets).
    """
    _ = k_closest  # Kept for API compatibility with notebook calls.
    edges = []
    platforms = nodes["platform"].unique().tolist()

    platform_to_idx = {
        p: nodes.index[nodes["platform"] == p].to_numpy()
        for p in platforms
    }
    platform_to_years = {p: nodes.loc[idxs, "year"].to_numpy() for p, idxs in platform_to_idx.items()}

    for target_idx, target_row in tqdm(nodes.iterrows(), total=len(nodes), desc="Building influence edges (first occurrence)"):
        target_year = int(target_row["year"])
        target_platform = target_row["platform"]

        candidate_rows = []

        for source_platform in platforms:
            if source_platform == target_platform:
                continue

            src_idxs_all = platform_to_idx[source_platform]
            src_years_all = platform_to_years[source_platform]

            valid_mask = src_years_all <= target_year
            if max_lag is not None:
                valid_mask = valid_mask & ((target_year - src_years_all) <= max_lag)
            if not np.any(valid_mask):
                continue

            src_idxs = src_idxs_all[valid_mask]
            src_years = src_years_all[valid_mask]
            lags = target_year - src_years

            sims = emb[src_idxs] @ emb[target_idx]
            keep = sims >= sim_threshold
            if not np.any(keep):
                continue

            for s_idx, s_year, lag, sim in zip(src_idxs[keep], src_years[keep], lags[keep], sims[keep]):
                candidate_rows.append({
                    "source": int(s_idx),
                    "target": int(target_idx),
                    "similarity": float(sim),
                    "lag_years": int(lag),
                    "source_platform": nodes.loc[s_idx, "platform"],
                    "target_platform": target_platform,
                    "source_year": int(s_year),
                    "target_year": target_year,
                    "source_sentence": nodes.loc[s_idx, "sentence"],
                    "target_sentence": target_row["sentence"],
                })

        if candidate_rows:
            cand_df = pd.DataFrame(candidate_rows)
            # First-occurrence attribution rule: earliest source_year wins;
            # tie-breakers prefer higher similarity, then lower source index.
            chosen = cand_df.sort_values(
                by=["source_year", "similarity", "source"],
                ascending=[True, False, True]
            ).iloc[0]
            edges.append(chosen.to_dict())

    return pd.DataFrame(edges)


def _build_within_platform_update_edges(nodes, emb, sim_threshold=0.88, max_lag=10):
    """
    Build within-platform carry-over edges: previous ToS update -> next update.

    This captures when the same platform reuses highly similar clause language
    in a later revision.
    """
    edges = []

    for platform, g in nodes.groupby("platform", sort=False):
        g = g.sort_values(["year", "node_id"]).reset_index(drop=True)
        ids = g["node_id"].to_numpy()
        years = g["year"].to_numpy()

        for pos in range(len(g)):
            target_idx = int(ids[pos])
            target_year = int(years[pos])

            prev_mask = years < target_year
            if not np.any(prev_mask):
                continue

            prev_ids = ids[prev_mask].astype(int)
            prev_years = years[prev_mask].astype(int)
            lags = target_year - prev_years

            if max_lag is not None:
                lag_ok = lags <= max_lag
                prev_ids = prev_ids[lag_ok]
                prev_years = prev_years[lag_ok]
                lags = lags[lag_ok]
                if len(prev_ids) == 0:
                    continue

            min_lag = lags.min()
            closest_ids = prev_ids[lags == min_lag]
            closest_years = prev_years[lags == min_lag]
            closest_lags = lags[lags == min_lag]

            sims = emb[closest_ids] @ emb[target_idx]
            keep = sims >= sim_threshold
            if not np.any(keep):
                continue

            cand = pd.DataFrame({
                "source": closest_ids[keep].astype(int),
                "target": target_idx,
                "similarity": sims[keep].astype(float),
                "lag_years": closest_lags[keep].astype(int),
                "source_platform": platform,
                "target_platform": platform,
                "source_year": closest_years[keep].astype(int),
                "target_year": target_year,
            })

            best = cand.sort_values(by=["similarity", "source"], ascending=[False, True]).iloc[0]
            source_idx = int(best["source"])
            edges.append({
                "source": source_idx,
                "target": target_idx,
                "similarity": float(best["similarity"]),
                "lag_years": int(best["lag_years"]),
                "source_platform": platform,
                "target_platform": platform,
                "source_year": int(best["source_year"]),
                "target_year": target_year,
                "source_sentence": nodes.loc[source_idx, "sentence"],
                "target_sentence": nodes.loc[target_idx, "sentence"],
            })

    return pd.DataFrame(edges)


def _build_coordination_edges(nodes, emb, sim_threshold=0.90, time_window=1, k_closest=2):
    edges = []
    platforms = nodes["platform"].unique().tolist()

    platform_to_idx = {
        p: nodes.index[nodes["platform"] == p].to_numpy()
        for p in platforms
    }
    platform_to_years = {p: nodes.loc[idxs, "year"].to_numpy() for p, idxs in platform_to_idx.items()}

    for anchor_idx, anchor_row in tqdm(nodes.iterrows(), total=len(nodes), desc="Building coordination edges"):
        anchor_year = int(anchor_row["year"])
        anchor_platform = anchor_row["platform"]

        for other_platform in platforms:
            if other_platform == anchor_platform:
                continue

            cand_idxs_all = platform_to_idx[other_platform]
            cand_years_all = platform_to_years[other_platform]

            time_diff = np.abs(cand_years_all - anchor_year)
            valid_mask = time_diff <= time_window
            if not np.any(valid_mask):
                continue

            cand_idxs = cand_idxs_all[valid_mask]
            cand_time_diff = time_diff[valid_mask]

            order = np.argsort(cand_time_diff)
            cand_idxs = cand_idxs[order][:k_closest]

            sims = emb[cand_idxs] @ emb[anchor_idx]
            keep = sims >= sim_threshold

            for c_idx, sim in zip(cand_idxs[keep], sims[keep]):
                i, j = sorted([int(anchor_idx), int(c_idx)])
                edges.append({
                    "u": i,
                    "v": j,
                    "similarity": float(sim),
                    "u_platform": nodes.loc[i, "platform"],
                    "v_platform": nodes.loc[j, "platform"],
                    "u_year": int(nodes.loc[i, "year"]),
                    "v_year": int(nodes.loc[j, "year"]),
                })

    if not edges:
        return pd.DataFrame(columns=["u", "v", "similarity", "u_platform", "v_platform", "u_year", "v_year"])

    coord_df = pd.DataFrame(edges)
    coord_df = coord_df.sort_values("similarity", ascending=False).drop_duplicates(subset=["u", "v"])
    return coord_df.reset_index(drop=True)


def visualize_platform_adoption(source_platform, influence_edges_df, nodes_df, min_similarity=0.85):
    """
    Interactive visualization showing how a source platform's clauses are adopted by others.
    Hover over edges and nodes to see sentence content.
    """
    edges = influence_edges_df[influence_edges_df["source_platform"] == source_platform].copy()

    if edges.empty:
        print(f"No influence edges found for {source_platform}")
        return None

    edges = edges[edges["similarity"] >= min_similarity].copy()

    target_summary = (
        edges
        .groupby("target_platform")
        .agg({
            "similarity": ["count", "mean", "max"],
            "lag_years": "mean",
            "target_sentence": lambda x: "<br>".join(x.str.slice(0, 100).values[:3])
        })
        .reset_index()
    )
    target_summary.columns = ["platform", "n_adoptions", "mean_sim", "max_sim", "avg_lag", "sample_targets"]

    platforms_involved = [source_platform] + target_summary["platform"].tolist()

    n_targets = len(target_summary)
    angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)

    node_x = [0] + [2 * np.cos(a) for a in angles]
    node_y = [0] + [2 * np.sin(a) for a in angles]

    node_size = [80]
    node_size += [30 + 2 * n for n in target_summary["n_adoptions"].values]

    node_color = ["#FF6B6B"]
    node_color += ["#4ECDC4"] * n_targets

    node_text = [
        f"<b>{source_platform}</b><br>Source Platform<br>{len(edges)} outgoing clauses"
    ]
    for _, row in target_summary.iterrows():
        node_text.append(
            f"<b>{row['platform']}</b><br>"
            f"Adoptions: {int(row['n_adoptions'])}<br>"
            f"Avg Similarity: {row['mean_sim']:.3f}<br>"
            f"Avg Lag (years): {row['avg_lag']:.1f}"
        )

    edge_x = []
    edge_y = []
    edge_hover = []

    for idx, (_, row) in enumerate(target_summary.iterrows(), 1):
        target_platform = row["platform"]
        target_edges = edges[edges["target_platform"] == target_platform]

        edge_x.extend([0, node_x[idx], None])
        edge_y.extend([0, node_y[idx], None])

        hover_text = f"<b>{source_platform} → {target_platform}</b><br>"
        hover_text += f"Adoptions: {len(target_edges)}<br>"
        hover_text += f"Avg Similarity: {target_edges['similarity'].mean():.3f}<br>"
        hover_text += f"Avg Lag: {target_edges['lag_years'].mean():.1f} years<br><br>"
        hover_text += "<b>Sample source sentences:</b><br>"
        for src_sent in target_edges["source_sentence"].head(2).values:
            hover_text += f"• {src_sent[:120]}...<br>"
        hover_text += "<br><b>Sample target sentences:</b><br>"
        for tgt_sent in target_edges["target_sentence"].head(2).values:
            hover_text += f"• {tgt_sent[:120]}...<br>"

        edge_hover.append(hover_text)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color="white", width=2),
            opacity=0.9,
        ),
        text=platforms_involved,
        textposition="middle center",
        textfont=dict(size=10, color="white", family="Arial Black"),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    for idx, (_, row) in enumerate(target_summary.iterrows(), 1):
        target_edges = edges[edges["target_platform"] == row["platform"]]

        fig.add_trace(go.Scatter(
            x=[0, node_x[idx]],
            y=[0, node_y[idx]],
            mode="lines",
            line=dict(
                width=1 + min(5, len(target_edges) / 3),
                color=f"rgba({int(255 - 100 * target_edges['similarity'].mean())}, {int(150 + 50 * target_edges['similarity'].mean())}, 100, 0.6)",
            ),
            hovertext=edge_hover[idx - 1],
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        title=f"<b>How {source_platform.upper()} clauses spread to other platforms</b><br><sub>Node size = adoption count | Line color/width = similarity/adoptions</sub>",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(240, 240, 245, 0.9)",
        height=700,
        width=1000,
    )

    return fig


def visualize_sentence_similarity_timeline(sentence_idx, nodes_df, emb_norm, min_similarity=0.80):
    """
    Visualize how similar sentences spread across platforms over time.
    Returns figure and results dataframe.
    """
    if sentence_idx < 0 or sentence_idx >= len(nodes_df):
        return None, None

    root_row = nodes_df.iloc[sentence_idx]
    root_embedding = emb_norm[sentence_idx]
    root_platform = root_row["platform"]
    root_year = root_row["year"]
    root_sentence = root_row["sentence"]

    similarities = emb_norm @ root_embedding

    results = nodes_df.copy()
    results["similarity"] = similarities
    results = results[results["similarity"] >= min_similarity].copy()
    results = results.sort_values("similarity", ascending=False).reset_index(drop=True)

    if len(results) < 2:
        return None, results

    similar = results[results["node_id"] != sentence_idx].copy()

    fig = go.Figure()

    platforms_in_data = sorted(similar["platform"].unique().tolist())
    palette = sns.color_palette("husl", n_colors=len(platforms_in_data))
    color_map = {
        p: f"rgba({int(palette[i][0] * 255)}, {int(palette[i][1] * 255)}, {int(palette[i][2] * 255)}, 0.7)"
        for i, p in enumerate(platforms_in_data)
    }

    fig.add_trace(go.Scatter(
        x=[root_year],
        y=[1.0],
        mode="markers",
        marker=dict(size=15, color="#FF4444", symbol="star", line=dict(color="white", width=2)),
        name="Root",
        hovertext=f"<b>ROOT</b><br>{root_platform.upper()} ({int(root_year)})<br>Similarity: 1.000<br><br>{root_sentence[:180]}",
        hoverinfo="text",
        showlegend=True,
    ))

    for platform in platforms_in_data:
        platform_data = similar[similar["platform"] == platform]

        fig.add_trace(go.Scatter(
            x=platform_data["year"],
            y=platform_data["similarity"],
            mode="markers",
            marker=dict(
                size=6 + platform_data["similarity"] * 12,
                color=color_map[platform],
                line=dict(color="white", width=1),
                opacity=0.8,
            ),
            name=platform,
            hovertext=[
                f"<b>{p}</b> ({int(y)})<br>Similarity: {sim:.3f}<br><br>{sent[:200]}"
                for p, y, sim, sent in zip(
                    platform_data["platform"],
                    platform_data["year"],
                    platform_data["similarity"],
                    platform_data["sentence"],
                )
            ],
            hoverinfo="text",
        ))

    fig.add_hline(
        y=min_similarity,
        line_dash="dash",
        line_color="rgba(150,150,150,0.4)",
        annotation_text=f"Min: {min_similarity:.2f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"<b>Similarity Timeline: {root_platform.upper()} ({int(root_year)})</b><br><sub>{root_sentence[:100]}...</sub>",
        xaxis_title="Year",
        yaxis_title="Cosine Similarity",
        height=600,
        width=1200,
        hovermode="closest",
        plot_bgcolor="rgba(245, 245, 250, 0.9)",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)"),
        yaxis=dict(range=[min_similarity - 0.02, 1.02], showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)"),
    )

    return fig, results


def visualize_sentence_causal_timeline(sentence_idx, nodes_df, emb_norm, influence_edges_df, min_similarity=0.80):
    """
    Visualize only causally valid adoptions from a root sentence.

    A target sentence is shown only if there is an edge source=sentence_idx -> target
    in influence_edges_df. Similarity is recomputed from emb_norm for display/filtering.
    Returns figure and results dataframe.
    """
    if sentence_idx < 0 or sentence_idx >= len(nodes_df):
        return None, None

    if influence_edges_df is None or len(influence_edges_df) == 0:
        return None, pd.DataFrame()

    root_row = nodes_df.iloc[sentence_idx]
    root_embedding = emb_norm[sentence_idx]
    root_platform = root_row["platform"]
    root_year = int(root_row["year"])
    root_sentence = root_row["sentence"]

    outgoing = influence_edges_df[influence_edges_df["source"] == sentence_idx].copy()
    if outgoing.empty:
        return None, pd.DataFrame()

    candidate_ids = outgoing["target"].astype(int).tolist()
    candidate_df = nodes_df.iloc[candidate_ids].copy()
    candidate_df["similarity"] = emb_norm[candidate_ids] @ root_embedding
    candidate_df = candidate_df[candidate_df["similarity"] >= min_similarity].copy()

    if candidate_df.empty:
        return None, candidate_df

    candidate_df = candidate_df.sort_values(["year", "similarity"], ascending=[True, False]).reset_index(drop=True)

    fig = go.Figure()

    platforms_in_data = sorted(candidate_df["platform"].unique().tolist())
    palette = sns.color_palette("husl", n_colors=max(1, len(platforms_in_data)))
    color_map = {
        p: f"rgba({int(palette[i][0] * 255)}, {int(palette[i][1] * 255)}, {int(palette[i][2] * 255)}, 0.80)"
        for i, p in enumerate(platforms_in_data)
    }

    fig.add_trace(go.Scatter(
        x=[root_year],
        y=[1.0],
        mode="markers",
        marker=dict(size=15, color="#FF4444", symbol="star", line=dict(color="white", width=2)),
        name="Root",
        hovertext=f"<b>ROOT</b><br>{root_platform.upper()} ({root_year})<br>Similarity: 1.000<br><br>{root_sentence[:180]}",
        hoverinfo="text",
        showlegend=True,
    ))

    for platform in platforms_in_data:
        platform_data = candidate_df[candidate_df["platform"] == platform]

        fig.add_trace(go.Scatter(
            x=platform_data["year"],
            y=platform_data["similarity"],
            mode="markers",
            marker=dict(
                size=7 + platform_data["similarity"] * 10,
                color=color_map[platform],
                line=dict(color="white", width=1),
                opacity=0.9,
            ),
            name=platform,
            hovertext=[
                f"<b>{p}</b> ({int(y)})<br>Causal edge from root: Yes<br>Similarity: {sim:.3f}<br><br>{sent[:200]}"
                for p, y, sim, sent in zip(
                    platform_data["platform"],
                    platform_data["year"],
                    platform_data["similarity"],
                    platform_data["sentence"],
                )
            ],
            hoverinfo="text",
        ))

    line_x = []
    line_y = []
    for _, row in candidate_df.iterrows():
        line_x.extend([root_year, int(row["year"]), None])
        line_y.extend([1.0, float(row["similarity"]), None])

    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.25)", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_hline(
        y=min_similarity,
        line_dash="dash",
        line_color="rgba(150,150,150,0.4)",
        annotation_text=f"Min: {min_similarity:.2f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"<b>Causal Adoption Timeline: {root_platform.upper()} ({root_year})</b><br><sub>Only targets with valid first-occurrence influence edges from the root are shown.</sub>",
        xaxis_title="Year",
        yaxis_title="Cosine Similarity",
        height=600,
        width=1200,
        hovermode="closest",
        plot_bgcolor="rgba(245, 245, 250, 0.9)",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)"),
        yaxis=dict(range=[max(0.0, min_similarity - 0.02), 1.02], showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)"),
    )

    return fig, candidate_df


def _check_temporal_direction(df, name):
    if df is None or len(df) == 0:
        print(f"{name}: empty")
        return
    required = {"source_year", "target_year"}
    if not required.issubset(df.columns):
        print(f"{name}: missing columns {required - set(df.columns)}")
        return

    bad = df[df["source_year"] > df["target_year"]].copy()
    same = (df["source_year"] == df["target_year"]).sum()
    forward = (df["source_year"] < df["target_year"]).sum()

    print(f"\n{name}")
    print(f"  total edges: {len(df):,}")
    print(f"  strictly forward (source < target): {forward:,}")
    print(f"  same-time (source == target): {same:,}")
    print(f"  backward violations (source > target): {len(bad):,}")

    if len(bad) > 0:
        print("  sample violations:")
        display(bad[["source", "target", "source_platform", "target_platform", "source_year", "target_year"]].head(10))


def _earliest_candidate_miss_rate(nodes, emb, influence_edges_df, sim_threshold, max_lag):
    miss_rows = []
    edge_lookup = influence_edges_df.set_index("target") if len(influence_edges_df) else pd.DataFrame()

    platforms = nodes["platform"].unique().tolist()
    platform_to_idx = {
        p: nodes.index[nodes["platform"] == p].to_numpy()
        for p in platforms
    }
    platform_to_years = {p: nodes.loc[idxs, "year"].to_numpy() for p, idxs in platform_to_idx.items()}

    for target_idx, target_row in nodes.iterrows():
        target_idx = int(target_idx)
        if len(influence_edges_df) == 0 or target_idx not in edge_lookup.index:
            continue

        target_year = int(target_row["year"])
        target_platform = target_row["platform"]

        all_candidates = []
        for source_platform in platforms:
            if source_platform == target_platform:
                continue

            src_idxs_all = platform_to_idx[source_platform]
            src_years_all = platform_to_years[source_platform]

            valid_mask = src_years_all <= target_year
            if max_lag is not None:
                valid_mask = valid_mask & ((target_year - src_years_all) <= max_lag)
            if not np.any(valid_mask):
                continue

            src_idxs = src_idxs_all[valid_mask]
            src_years = src_years_all[valid_mask]
            sims = emb[src_idxs] @ emb[target_idx]
            keep = sims >= sim_threshold

            for s_idx, s_year, sim in zip(src_idxs[keep], src_years[keep], sims[keep]):
                all_candidates.append((int(s_idx), int(s_year), float(sim)))

        if not all_candidates:
            continue

        earliest_year = min(y for _, y, _ in all_candidates)
        chosen = edge_lookup.loc[target_idx]
        if isinstance(chosen, pd.DataFrame):
            chosen = chosen.iloc[0]
        chosen_year = int(chosen["source_year"])

        if chosen_year != earliest_year:
            miss_rows.append({
                "target": target_idx,
                "chosen_source": int(chosen["source"]),
                "chosen_source_year": chosen_year,
                "earliest_valid_source_year": int(earliest_year),
                "target_year": target_year,
                "target_platform": target_platform,
            })

    miss_df = pd.DataFrame(miss_rows)
    total_targets = len(influence_edges_df["target"].unique()) if len(influence_edges_df) else 0
    miss_rate = (len(miss_df) / total_targets) if total_targets else 0.0
    return miss_df, total_targets, miss_rate
