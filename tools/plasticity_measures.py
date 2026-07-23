import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def assignment_based_effort(X: np.ndarray, Y: np.ndarray, similarity_thresh: float = 0.7) -> dict:
    """
    Estimate reading effort using one-to-one optimal clause assignment
    rather than nearest-neighbor matching.

    Rationale
    ---------
    Nearest-neighbor alignment allows many-to-one matches: e.g. two
    different Y clauses can both claim the same X clause as their
    "closest match," silently hiding a clause split (one original
    obligation broken into two new ones) or clause merge. A reader
    doing a redline review experiences a split/merge as extra effort
    (new structure to parse), which many-to-one matching underestimates.

    This uses the Hungarian algorithm to find the best one-to-one
    pairing between clauses (up to min(n, m) pairs), then treats any
    leftover, unmatched clauses on either side as pure "new reading"
    or "pure deletion" -- since by definition they had no available
    partner left to be paired with.

    Returns
    -------
    dict with 'total_effort', 'unmatched_X' (deleted clause indices),
    'unmatched_Y' (new clause indices), and 'matched_pairs' (index pairs
    with their similarity).
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    n, m = len(X), len(Y)

    C = cdist(X, Y, metric="cosine")  # cost matrix, shape (n, m)
    row_ind, col_ind = linear_sum_assignment(C)

    matched_pairs = []
    effort = 0.0
    for i, j in zip(row_ind, col_ind):
        dist = C[i, j]
        sim = 1 - dist
        if sim < similarity_thresh:
            # Even the "best available" partner is too different to
            # count as the same clause -- treat as effectively unmatched
            continue
        matched_pairs.append((i, j, sim))
        effort += dist  # only the "modified" gap contributes

    matched_x = {i for i, _, _ in matched_pairs}
    matched_y = {j for _, j, _ in matched_pairs}
    unmatched_x = [i for i in range(n) if i not in matched_x]
    unmatched_y = [j for j in range(m) if j not in matched_y]

    # Unmatched clauses cost full read/verification effort each
    effort += len(unmatched_x) * 1.0
    effort += len(unmatched_y) * 1.0

    return {
        "total_effort": float(effort),
        "unmatched_X": unmatched_x,   # likely deletions or split sources
        "unmatched_Y": unmatched_y,   # likely new clauses or split results
        "matched_pairs": matched_pairs,
    }

import numpy as np
from scipy.spatial.distance import cdist


def contract_revision_effort(
    X: np.ndarray,
    Y: np.ndarray,
    unchanged_thresh: float = 0.05,
    new_clause_thresh: float = 0.5,
    new_clause_weight: float = 2.0,
    deleted_clause_weight: float = 1.5,
) -> dict:
    """
    Estimate the reading effort required to review an updated contract (Y)
    given that the reader already knows the original version (X).

    Rationale
    ---------
    Unlike reading two unrelated documents, a contract revision is mostly
    the SAME text with a few localized edits. A reader who already knows X
    doesn't need to re-read unchanged clauses at all -- their effort is
    concentrated on:
      1. Clauses that were reworded/modified (moderate effort: has to
         carefully compare new phrasing against what they remember).
      2. Clauses that are entirely NEW in Y, with no counterpart in X
         (high effort: nothing to anchor against, must be read from
         scratch and assessed independently).
      3. Clauses that were DELETED from X (i.e. present in the original
         but with no counterpart in Y at all) -- also high effort, since
         a careful reader must notice the absence and consider its legal
         implications (e.g. "wait, where did the indemnification clause
         go?"). This is easy to miss and arguably as risky as a new clause.
      4. Clauses that are unchanged (near-zero effort: can be skimmed or
         skipped, since the reader already knows them, even if their
         position in the document moved).

    This function performs symmetric nearest-neighbor alignment: each
    clause in Y is matched to its closest clause in X (to detect
    unchanged/modified/new), AND each clause in X is matched to its
    closest clause in Y (to detect deletions). Both contribute to the
    total weighted effort score.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Sentence/clause embeddings for the original contract.
    Y : np.ndarray, shape (m, d)
        Sentence/clause embeddings for the updated contract.
    unchanged_thresh : float
        Cosine distance below which a clause is considered unchanged
        (i.e. effectively identical to its nearest counterpart).
    new_clause_thresh : float
        Cosine distance above which a Y clause is considered entirely new
        (no meaningful counterpart in X), or an X clause is considered
        entirely deleted (no meaningful counterpart in Y).
    new_clause_weight : float
        Multiplier applied to new clauses' effort, reflecting the extra
        cognitive cost of clauses with no anchor in prior knowledge.
    deleted_clause_weight : float
        Multiplier applied to deleted clauses' effort, reflecting the
        cost of noticing and evaluating an absence. Typically slightly
        lower than new_clause_weight, since confirming a deletion is
        usually faster than reading and assessing brand-new text -- but
        this is a modeling choice and can be tuned.

    Returns
    -------
    dict with:
        'total_effort'        : float, overall weighted reading-effort score
        'per_clause_effort_Y' : np.ndarray (m,), effort per Y clause (unchanged/modified/new)
        'labels_Y'            : list[str] of 'unchanged' | 'modified' | 'new' per Y clause
        'deleted_effort_X'    : np.ndarray (n,), effort per X clause (0 if retained, weighted if deleted)
        'labels_X'            : list[str] of 'retained' | 'deleted' per X clause
        'pct_unchanged'       : float, fraction of Y clauses unchanged
        'pct_modified'        : float, fraction of Y clauses modified
        'pct_new'             : float, fraction of Y clauses entirely new
        'pct_deleted'         : float, fraction of X clauses deleted (relative to len(X))
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)

    # --- Forward pass: for each Y clause, find nearest X clause ---
    D_yx = cdist(Y, X, metric="cosine")        # shape (m, n)
    nearest_dist_y = D_yx.min(axis=1)

    labels_y = []
    per_clause_effort_y = np.zeros(len(Y))

    for i, dist in enumerate(nearest_dist_y):
        if dist <= unchanged_thresh:
            labels_y.append("unchanged")
            per_clause_effort_y[i] = 0.0
        elif dist >= new_clause_thresh:
            labels_y.append("new")
            per_clause_effort_y[i] = new_clause_weight * dist
        else:
            labels_y.append("modified")
            per_clause_effort_y[i] = dist

    # --- Backward pass: for each X clause, find nearest Y clause ---
    # (symmetric deletion detection -- a clause "disappears" if nothing
    # in Y is close enough to count as its surviving counterpart)
    D_xy = cdist(X, Y, metric="cosine")         # shape (n, m)
    nearest_dist_x = D_xy.min(axis=1)

    labels_x = []
    deleted_effort_x = np.zeros(len(X))

    for i, dist in enumerate(nearest_dist_x):
        if dist >= new_clause_thresh:
            labels_x.append("deleted")
            deleted_effort_x[i] = deleted_clause_weight * dist
        else:
            labels_x.append("retained")
            deleted_effort_x[i] = 0.0

    total_effort = per_clause_effort_y.sum() + deleted_effort_x.sum()

    m, n = len(Y), len(X)

    return {
        "total_effort": float(total_effort),
        "per_clause_effort_Y": per_clause_effort_y,
        "labels_Y": labels_y,
        "deleted_effort_X": deleted_effort_x,
        "labels_X": labels_x,
        "pct_unchanged": labels_y.count("unchanged") / m,
        "pct_modified": labels_y.count("modified") / m,
        "pct_new": labels_y.count("new") / m,
        "pct_deleted": labels_x.count("deleted") / n,
    }

def risk_weighted_effort(
    X: np.ndarray,
    Y: np.ndarray,
    risk_exemplars: np.ndarray,
    unchanged_thresh: float = 0.07,
    base_risk_weight: float = 1.0,
    max_risk_weight: float = 3.0,
) -> dict:
    """
    Estimate reading effort where each Y clause's change-cost is scaled
    by how legally consequential that clause appears to be.

    Rationale
    ---------
    A reader doesn't allocate attention uniformly. A reworded notice
    address is low-stakes even if flagged as "modified"; a reworded
    liability cap or indemnification clause is high-stakes even if the
    edit distance is small. This weights modified/new clause effort by
    similarity to a small reference set of known risk-relevant clause
    embeddings (e.g. embeddings of template indemnification, liability,
    termination, IP-assignment clauses), so structurally similar changes
    near those "risk zones" get amplified effort scores.

    Parameters
    ----------
    risk_exemplars : np.ndarray, shape (k, d)
        Embeddings of reference clauses representing high-risk categories
        (indemnification, liability, termination, etc.). You supply these
        -- e.g. embed a handful of clauses you already know are
        high-stakes.

    Returns
    -------
    dict with 'total_effort' and 'per_clause_effort' (weighted by risk proximity).
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    risk_exemplars = np.atleast_2d(risk_exemplars)

    D_yx = cdist(Y, X, metric="cosine")
    nearest_dist = D_yx.min(axis=1)

    # How close is each Y clause to a "risky" clause type?
    risk_sim = 1 - cdist(Y, risk_exemplars, metric="cosine").min(axis=1)
    risk_sim = np.clip(risk_sim, 0, 1)
    risk_weight = base_risk_weight + risk_sim * (max_risk_weight - base_risk_weight)

    per_clause_effort = np.where(
        nearest_dist <= unchanged_thresh,
        0.0,
        nearest_dist * risk_weight,
    )

    return {
        "total_effort": float(per_clause_effort.sum()),
        "per_clause_effort": per_clause_effort,
        "risk_weight": risk_weight,
    }

def context_disruption_effort(X: np.ndarray, Y: np.ndarray, window: int = 2) -> dict:
    """
    Estimate reading effort by detecting clauses whose LOCAL similarity
    context changed, even if the clause's own nearest match is stable.

    Rationale
    ---------
    Some edits are self-contained (e.g. fixing a typo). Others force a
    careful reader to re-examine surrounding clauses too -- e.g. a
    redefinition of a key term changes how every clause referencing it
    should be (re-)read, even if none of those clauses' own text
    changed. This is approximated by comparing each Y clause's
    similarity profile against its own local neighborhood (+/- window)
    to the equivalent profile in X: a big shift signals the clause's
    "meaning in context" changed even if its literal wording didn't.

    Returns
    -------
    dict with 'total_effort' and 'per_clause_disruption' (per Y clause index).
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    n, m = len(X), len(Y)

    D_yx = cdist(Y, X, metric="cosine")
    nearest_idx = D_yx.argmin(axis=1)  # best X match for each Y clause

    disruption = np.zeros(m)
    for j in range(m):
        i = nearest_idx[j]

        # Local neighborhood of Y clause j (within Y)
        y_lo, y_hi = max(0, j - window), min(m, j + window + 1)
        y_neighbors = Y[y_lo:y_hi]

        # Local neighborhood of its matched X clause (within X)
        x_lo, x_hi = max(0, i - window), min(n, i + window + 1)
        x_neighbors = X[x_lo:x_hi]

        # Compare how similar clause j is to ITS OWN neighbors in Y
        # vs how similar clause i was to ITS OWN neighbors in X
        y_local_sim = 1 - cdist(Y[j:j+1], y_neighbors, metric="cosine").mean()
        x_local_sim = 1 - cdist(X[i:i+1], x_neighbors, metric="cosine").mean()

        disruption[j] = abs(y_local_sim - x_local_sim)

    return {
        "total_effort": float(disruption.sum()),
        "per_clause_disruption": disruption,
    }

import numpy as np
import torch
from scipy.spatial.distance import cdist
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_lm(model_name: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def _get_tokenizer_max_length(tokenizer) -> int:
    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or max_length <= 0 or max_length > 100000:
        return 1024
    return int(max_length)


def sentence_perplexity(sentences, model, tokenizer, max_perplexity=1000.0):
    perplexities = np.zeros(len(sentences))
    max_length = _get_tokenizer_max_length(tokenizer)
    with torch.no_grad():
        for i, sent in enumerate(sentences):
            if not sent.strip():
                perplexities[i] = 0.0
                continue
            inputs = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexities[i] = torch.exp(outputs.loss).item()
    return np.clip(perplexities, 0.0, max_perplexity)


def document_perplexity(sentences, model, tokenizer, max_perplexity=1000.0):
    text = " ".join(s.strip() for s in sentences if s.strip())
    max_length = _get_tokenizer_max_length(tokenizer)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = model(**inputs, labels=inputs["input_ids"])
    perplexity = torch.exp(outputs.loss).item()
    return float(np.clip(perplexity, 0.0, max_perplexity))


def contract_reading_effort_density(
    X_texts: list[str],
    X_embeddings: np.ndarray,
    Y_texts: list[str],
    Y_embeddings: np.ndarray,
    new_clause_thresh: float = 0.3,
    new_clause_weight: float = 2.0,
    deleted_clause_weight: float = 1.5,
    model_name: str = "gpt2",
    max_perplexity: float = 1000.0,
) -> dict:
    """
    Estimate reading effort of updating X -> Y, normalized so the score
    reflects INTENSITY of difficulty/change rather than raw document
    length or clause count.

    Rationale
    ---------
    Summing perplexity (or any per-sentence cost) over all sentences
    means a longer document -- even one that is proportionally no
    harder and no more heavily revised -- will always score as "more
    effort" than a shorter one, simply because it has more sentences to
    sum over. That conflates volume with difficulty. This version uses
    two length-robust aggregations instead:

      1. MEAN (not sum) perplexity for the baseline document effort --
         representing "how hard is a typical sentence to read," which
         doesn't grow just because the document has more sentences.
      2. PROPORTIONAL added/deleted effort -- the added/deleted terms
         are the mean perplexity of added/deleted sentences, scaled by
         the FRACTION of the document they represent (not their raw
         count). A contract where 3 out of 10 clauses changed and one
         where 30 out of 100 changed (same 30% proportion) now score
         comparably, whereas raw counts would score the second 10x higher
         despite equivalent relative disruption.

    This makes the score comparable across contracts of very different
    lengths -- answering "how intensely does this revision demand
    attention, per unit of reading" rather than "how much total reading
    is there."

    Returns
    -------
    dict with:
        'baseline_perplexity'   : float, MEAN perplexity across X's sentences
        'added_effort_density'  : float, mean perplexity of added sentences,
                                   weighted by their fraction of Y
        'deleted_effort_density': float, mean perplexity of deleted sentences,
                                   weighted by their fraction of X
        'total_effort_density'  : float, sum of the three (length-normalized) terms
        'pct_added'             : float, fraction of Y that is new
        'pct_deleted'           : float, fraction of X that was removed
    """
    model, tokenizer = _load_lm(model_name)
    X_embeddings, Y_embeddings = np.atleast_2d(X_embeddings), np.atleast_2d(Y_embeddings)
    n, m = len(X_texts), len(Y_texts)

    # --- 1. Baseline: MEAN per-sentence perplexity of X, not a document-level sum ---
    # (mean over individually-scored sentences rather than one long concatenated
    # pass, so a longer X doesn't get a structurally different/longer context window)
    X_sentence_perplexities = sentence_perplexity(X_texts, model, tokenizer, max_perplexity)
    baseline_perplexity = float(X_sentence_perplexities.mean()) if n > 0 else 0.0

    # --- 2. Detect added / deleted sentences (same alignment logic as before) ---
    D_yx = cdist(Y_embeddings, X_embeddings, metric="cosine")
    added_mask = D_yx.min(axis=1) >= new_clause_thresh
    added_sentences = [Y_texts[j] for j in range(m) if added_mask[j]]

    D_xy = cdist(X_embeddings, Y_embeddings, metric="cosine")
    deleted_mask = D_xy.min(axis=1) >= new_clause_thresh
    deleted_sentences = [X_texts[i] for i in range(n) if deleted_mask[i]]

    pct_added = len(added_sentences) / m if m > 0 else 0.0
    pct_deleted = len(deleted_sentences) / n if n > 0 else 0.0

    # --- 3. Effort density: MEAN perplexity of changed sentences, scaled by
    #         PROPORTION of the document they represent (not raw count) ---
    added_perplexities = (
        sentence_perplexity(added_sentences, model, tokenizer, max_perplexity)
        if added_sentences else np.array([0.0])
    )
    deleted_perplexities = (
        sentence_perplexity(deleted_sentences, model, tokenizer, max_perplexity)
        if deleted_sentences else np.array([0.0])
    )

    added_effort_density = new_clause_weight * pct_added * added_perplexities.mean()
    deleted_effort_density = deleted_clause_weight * pct_deleted * deleted_perplexities.mean()

    total_effort_density = baseline_perplexity + added_effort_density + deleted_effort_density

    return {
        "baseline_perplexity": baseline_perplexity,
        "added_effort_density": float(added_effort_density),
        "deleted_effort_density": float(deleted_effort_density),
        "total_effort_density": float(total_effort_density),
        "pct_added": float(pct_added),
        "pct_deleted": float(pct_deleted),
    }

# import numpy as np
# import torch
# from scipy.spatial.distance import cdist
# from transformers import AutoModelForCausalLM, AutoTokenizer


# def _load_lm(model_name: str = "gpt2"):
#     """Load a causal LM + tokenizer once, for perplexity scoring."""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.eval()
#     return model, tokenizer


# def sentence_perplexity(
#     sentences: list[str],
#     model,
#     tokenizer,
#     max_perplexity: float = 1000.0,
# ) -> np.ndarray:
#     """
#     Compute the perplexity of each sentence individually under a causal LM,
#     clipped to a maximum value to guard against outliers.

#     Rationale
#     ---------
#     Perplexity is unbounded and can spike to extreme values (thousands
#     or more) on short sentences, rare proper nouns, unusual
#     tokenization, or malformed text -- none of which necessarily
#     reflects genuine reading difficulty, just a quirk of the language
#     model's token distribution. Left unclipped, a single such sentence
#     could dominate an aggregate effort score far out of proportion to
#     its actual contribution to reading effort. Clipping caps this
#     influence while still preserving the relative ordering of
#     "genuinely hard to read" vs "easy" for the vast majority of
#     sentences that fall well within normal range.

#     Parameters
#     ----------
#     max_perplexity : float
#         Upper bound; any sentence perplexity above this is clipped down
#         to it. Tune based on the LM used -- e.g. GPT-2 perplexities on
#         normal English prose are typically well under 200-300; a cap
#         around 1000 catches only genuine outliers rather than
#         compressing meaningful variation.

#     Returns
#     -------
#     np.ndarray, shape (len(sentences),) -- clipped perplexity of each sentence.
#     """
#     perplexities = np.zeros(len(sentences))
#     with torch.no_grad():
#         for i, sent in enumerate(sentences):
#             if not sent.strip():
#                 perplexities[i] = 0.0
#                 continue
#             inputs = tokenizer(sent, return_tensors="pt")
#             outputs = model(**inputs, labels=inputs["input_ids"])
#             perplexities[i] = torch.exp(outputs.loss).item()

#     return np.clip(perplexities, 0.0, max_perplexity)


# def document_perplexity(
#     sentences: list[str],
#     model,
#     tokenizer,
#     max_perplexity: float = 1000.0,
# ) -> float:
#     """
#     Compute the perplexity of a whole document (all sentences concatenated),
#     clipped to guard against extreme values, representing the baseline
#     reading effort of that text on its own.
#     """
#     text = " ".join(s.strip() for s in sentences if s.strip())
#     with torch.no_grad():
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
#         outputs = model(**inputs, labels=inputs["input_ids"])
#     perplexity = torch.exp(outputs.loss).item()
#     return float(np.clip(perplexity, 0.0, max_perplexity))


# def contract_reading_effort_with_perplexity(
#     X_texts: list[str],
#     X_embeddings: np.ndarray,
#     Y_texts: list[str],
#     Y_embeddings: np.ndarray,
#     unchanged_thresh: float = 0.07,
#     new_clause_thresh: float = 0.3,
#     new_clause_weight: float = 2.0,
#     deleted_clause_weight: float = 1.5,
#     model_name: str = "gpt2",
#     max_perplexity: float = 1000.0,
# ) -> dict:
#     """
#     Estimate the total reading effort of moving from contract X to its
#     updated version Y, combining:

#       1. The BASELINE effort of reading X itself -- approximated by X's
#          document-level perplexity under a language model.
#       2. The EXTRA effort introduced specifically by the revision --
#          approximated by the perplexity of added and deleted sentences
#          (identified via embedding similarity).

#     All perplexity values (document-level and sentence-level) are
#     clipped to `max_perplexity` before aggregation, so that a single
#     outlier sentence (e.g. one with unusual tokenization or a rare
#     proper noun) cannot dominate the total score out of proportion to
#     its actual contribution to reading effort.

#     Parameters
#     ----------
#     max_perplexity : float
#         Upper bound applied to every individual perplexity value
#         (document-level and sentence-level) before summing. Tune based
#         on the LM used -- GPT-2 perplexities on normal prose are
#         typically well under a few hundred, so a cap around 1000
#         catches genuine outliers without compressing normal variation.

#     Returns
#     -------
#     dict with:
#         'baseline_perplexity'  : float, clipped perplexity of X as a whole document
#         'added_perplexity'     : float, weighted sum of clipped perplexities of new Y sentences
#         'deleted_perplexity'   : float, weighted sum of clipped perplexities of deleted X sentences
#         'total_effort'         : float, sum of the three components
#         'added_sentences'      : list[str], the sentences classified as new
#         'deleted_sentences'    : list[str], the sentences classified as deleted
#         'n_added_clipped'      : int, number of added sentences that hit the clip ceiling
#         'n_deleted_clipped'    : int, number of deleted sentences that hit the clip ceiling
#     """
#     model, tokenizer = _load_lm(model_name)

#     X_embeddings, Y_embeddings = np.atleast_2d(X_embeddings), np.atleast_2d(Y_embeddings)

#     # --- 1. Baseline effort: perplexity of X as a whole ---
#     baseline_perplexity = document_perplexity(X_texts, model, tokenizer, max_perplexity)

#     # --- 2. Detect added sentences (Y clauses with no good match in X) ---
#     D_yx = cdist(Y_embeddings, X_embeddings, metric="cosine")
#     nearest_dist_y = D_yx.min(axis=1)
#     added_mask = nearest_dist_y >= new_clause_thresh
#     added_sentences = [Y_texts[j] for j in range(len(Y_texts)) if added_mask[j]]

#     # --- 3. Detect deleted sentences (X clauses with no good match in Y) ---
#     D_xy = cdist(X_embeddings, Y_embeddings, metric="cosine")
#     nearest_dist_x = D_xy.min(axis=1)
#     deleted_mask = nearest_dist_x >= new_clause_thresh
#     deleted_sentences = [X_texts[i] for i in range(len(X_texts)) if deleted_mask[i]]

#     # --- 4. Clipped perplexity of just the added / deleted sentences ---
#     added_perplexities = (
#         sentence_perplexity(added_sentences, model, tokenizer, max_perplexity)
#         if added_sentences else np.array([0.0])
#     )
#     deleted_perplexities = (
#         sentence_perplexity(deleted_sentences, model, tokenizer, max_perplexity)
#         if deleted_sentences else np.array([0.0])
#     )

#     n_added_clipped = int(np.sum(added_perplexities >= max_perplexity))
#     n_deleted_clipped = int(np.sum(deleted_perplexities >= max_perplexity))

#     added_perplexity = new_clause_weight * added_perplexities.sum()
#     deleted_perplexity = deleted_clause_weight * deleted_perplexities.sum()

#     total_effort = baseline_perplexity + added_perplexity + deleted_perplexity

#     return {
#         "baseline_perplexity": float(baseline_perplexity),
#         "added_perplexity": float(added_perplexity),
#         "deleted_perplexity": float(deleted_perplexity),
#         "total_effort": float(total_effort),
#         "added_sentences": added_sentences,
#         "deleted_sentences": deleted_sentences,
#         "n_added_clipped": n_added_clipped,
#         "n_deleted_clipped": n_deleted_clipped,
#     }