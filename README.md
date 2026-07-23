
# Plasticity
Code for analysing plasticity and text reuse in terms of use

## Plasticity Analysis Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kasparvonbeelen/contracts/blob/main/3-1-influence_analysis.ipynb)


This notebook analyzes how contract clauses spread across platforms over time, using clause embeddings and a directed influence network. It builds similarity-based links between clauses, combines cross-platform influence with within-platform carry-over, and computes a recursive influence score that attributes downstream impact back to original root clauses.

The workflow then provides interactive exploration of influence chains, including:

root-based recursive chain visualization
node-level hover details with truncated sentence previews
year-based left-to-right layouts
platform-colored nodes for easier pattern recognition
In addition to sentence-level analysis, the notebook aggregates influence at platform and year levels, highlights top influential clauses and platforms, and produces coordination views showing when platforms adopt similar language in the same period. It is designed for both quantitative scoring and interpretable visual inspection of clause diffusion dynamics.

## Plasticity Measures

The measures in plasticity_measures.py are all trying to answer a related question: how much effort does it take to understand a changed contract compared with an older version? They do that in different ways, depending on what kind of “effort” you care about.

Plain-English Summary

### assignment_based_effort

This measure tries to match clauses in the old contract to clauses in the new contract one-to-one. It uses an optimization step so that one old clause cannot be reused as the “best match” for many new clauses.

In simple terms: it asks, “If I force each old clause to line up with at most one new clause, how much text looks changed, and how many clauses seem to have been added or removed?”

What it counts:

Small mismatch between matched clauses adds a little effort.
Clauses left unmatched are treated as full extra effort.
Why this matters:

It is good at noticing clause splits and merges, where one clause becomes two, or several clauses get collapsed into one.

### contract_revision_effort

This is the most direct “reviewing a revised contract” measure. It assumes the reader already knows the old contract, so unchanged clauses cost almost nothing, modified clauses cost some effort, new clauses cost more, and deleted clauses also cost effort because the reader has to notice something disappeared.

In simple terms: it asks, “If I already know version X, how hard is it to review version Y?”

What it counts:

Unchanged clause: near zero effort.
Modified clause: moderate effort.
New clause: high effort.
Deleted clause: also high effort, but weighted separately.
Why this matters:

It reflects a realistic redline-review situation better than a plain document similarity score.

## risk_weighted_effort

This measure starts from the same idea as change detection, but it gives more weight to clauses that look legally important. It compares each changed clause to a set of “risk exemplar” clauses, such as liability or indemnity examples.

In simple terms: it asks, “How much effort should I spend on this change, given that some kinds of clauses matter more than others?”

What it counts:

Changed clauses near high-risk clause types get amplified.
Unchanged clauses still get near zero effort.
Why this matters:

Not all edits are equally important. A tiny edit in a liability clause may deserve more attention than a larger edit in a mailing address clause.

### context_disruption_effort

This one looks beyond the clause itself and asks whether its local context changed. A clause might still match closely to an old clause, but if the surrounding neighborhood of clauses changes, the reader may need to reinterpret it.

In simple terms: it asks, “Even if this clause looks similar, has its meaning-in-context shifted because nearby clauses changed?”

What it counts:

It measures how much each clause’s relationship to nearby clauses has changed.
Why this matters:

It can catch structural or contextual changes that simple clause-to-clause matching misses.

### contract_reading_effort_density

This measure mixes change detection with language-model perplexity, which is used here as a rough proxy for how hard text is to read. It then normalizes the score so long contracts are not automatically judged as harder just because they contain more clauses.

In simple terms: it asks, “How intense is the reading burden of this revision per unit of text, rather than in total?”

What it counts:

Baseline reading difficulty of the old contract.
Extra difficulty from added clauses.
Extra difficulty from deleted clauses.
It scales added and deleted effort by their proportion of the document, not just raw count.
Why this matters:

It is designed for comparison across contracts of very different lengths.

## How They Differ

These measures differ mainly in what they treat as “effort”:

assignment_based_effort focuses on structural matching between clauses.
contract_revision_effort focuses on human review effort for unchanged, modified, new, and deleted clauses.
risk_weighted_effort focuses on importance, not just amount of change.
context_disruption_effort focuses on surrounding context and meaning shifts.
contract_reading_effort_density focuses on normalized reading burden and cross-document comparability.
Another way to see it:

If you care about whether clauses were split, merged, added, or removed, assignment_based_effort is the structural one.
If you care about practical redline review, contract_revision_effort is the most intuitive baseline.
If you care about legal stakes, risk_weighted_effort is the priority-sensitive version.
If you care about ripple effects from nearby clauses, context_disruption_effort is the contextual one.
If you care about comparing contracts of different sizes fairly, contract_reading_effort_density is the normalized reading-difficulty one.
Strengths And Weaknesses

### assignment_based_effort

Strength: good at avoiding misleading many-to-one matches.
Strength: better than nearest-neighbor matching for splits and merges.
Weakness: still reduces the problem to clause pairing, so it may miss broader context.
Weakness: unmatched clauses are treated in a fairly blunt all-or-nothing way.

### contract_revision_effort

Strength: easiest to interpret for contract review.
Strength: distinguishes unchanged, modified, new, and deleted clauses clearly.
Weakness: depends heavily on threshold choices.
Weakness: nearest-neighbor logic can still oversimplify more complex reorganizations.
### risk_weighted_effort

Strength: closer to how lawyers or reviewers actually prioritize attention.
Strength: recognizes that some clauses matter more than others.
Weakness: only as good as the chosen risk exemplars.
Weakness: can bias results toward whatever risk categories were supplied.

### context_disruption_effort

Strength: captures subtle changes in surrounding meaning.
Strength: useful when clause order or neighboring material matters.
Weakness: harder to explain and validate.
Weakness: can be sensitive to local window size and document structure.
### contract_reading_effort_density

Strength: more comparable across short and long contracts.
Strength: separates document length from revision intensity.
Weakness: perplexity is only an indirect proxy for human reading difficulty.
Weakness: depends on the language model and may reflect model oddities rather than real legal complexity.

## Bottom Line

If you want one general-purpose measure, contract_revision_effort is the clearest and most practical starting point. If you want to capture structural rewrites, use assignment_based_effort. If legal importance matters, use risk_weighted_effort. If you think edits change the meaning of nearby text, use context_disruption_effort. If you need fair comparison across contracts of different lengths, use contract_reading_effort_density.

The main tradeoff is simple: the more realistic the measure becomes, the more assumptions it introduces. Simpler measures are easier to explain but miss nuance; richer measures capture more of real review effort but depend more on thresholds, exemplars, and modeling choices.