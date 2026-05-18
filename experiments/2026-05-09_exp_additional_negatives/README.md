# Additional negative terms — diagram

Visualises which negative-pair structures are already in the contrastive
loss and which are candidates not yet covered, on a 2×3 (batch × time)
sub-window of the prediction graph. Each vertex is a positive pair
`(f_{b,τ-1} → h_{b,τ})`.

## Notation

- `f_{b,τ}` — forecast emitted at position `τ`, batch `b`. Trained to predict
  `h_{b,τ+1}`.
- `h_{b,τ}` — encoder embedding at position `τ`, batch `b`.
- Anchor `(b, s, c)` — the index over which one InfoNCE ratio is computed.
  Positive pair at the anchor: `(f_{b,s,c}, h_{b,s+1,c})`. Anchor index
  `s ∈ [0, T-2]`. Channel `c`.
- Top row in the diagram is batch `b`, bottom row is batch `b'` (with
  `b ≠ b'`).

## Two views of the same diagram

**Easier to read** — each candidate-new edge drawn once, at the centred
anchor `s = t-1`. One colour, one line:

![Centred view](plots/additional_negatives_diagram.png)

**More complete** — same diagram, but each colour now also drawn at every
translation copy that fits the 6-vertex window: anchor `s ± 1` plus the
`b ↔ b'` label-swap for cross-batch terms. The existing orange diag block
also gains its `b ↔ b'` mirror here, for the same reason.

![Echoes view](plots/additional_negatives_diagram_echoes.png)

A loss term is implemented per anchor `(b, s, c)` and summed over all anchors
and all ordered batch pairs `(b, b')` with `b ≠ b'`. Adding one colour to the
loss therefore creates many edges in this window — the centred view hides
that. The echoes view makes it explicit and shows where one source colour
physically overlaps with another (e.g. CB0 and CB1 are the same loss term
under `b ↔ b'`).

## Edges drawn in the diagram

### Existing (faded — already in `cosine_similarity_batch_square`, `src/loss.py:446-495`)

| Colour | Pair (centred at anchor `s = t-1`) | Loss term |
|---|---|---|
| green solid | `(f_{b,τ-1}, h_{b,τ})` — positive | numerator of every InfoNCE ratio |
| grey | within-b `(f_{b,τ}, f_{b,τ+1})` and `(h_{b,τ}, h_{b,τ+1})` — adj. `τ` | `neg_zy` (f-side) + `neg_xy` (h-side) |
| orange (faded) | cross-b `(h_{b',τ+1}, f_{b,τ})` — h leads f by 1 | `neg_cross_batch` |
| blue | cross-b `(f_{b,τ}, f_{b',τ})` same `τ` | `neg_cross_batch_forecast` |
| red | cross-b `(h_{b,τ+1}, h_{b',τ+1})` same `τ` | `neg_cross_batch_embedding` |

The within-b same-`τ` `(h_{b,τ}, f_{b,τ})` pair is also already in the loss as
the same-channel diagonal of `neg_xy_hat`
(`cos(h_{·,τ,c1}, f_{·,τ,c2})` summed over `c1` with no `c1 ≠ c2` mask). It
isn't drawn faded — only because that would require a new legend entry — but
it matters when reading the status of WB1 below. See the comment block at
`cosine_similarity_batch_add_neg_htft` in `src/loss.py:286-352` for the
explicit double-count analysis.

### Candidate-new (bright dashed)

Each row gives the pair structure as `(left, right)`; cosine similarity is
symmetric, so the order is cosmetic. The Status column says whether the pair
is genuinely uncovered by the existing loss, or whether some other term —
or just the cross-batch summation over `(b, b')` — already pulls it apart.

| Colour | Code | Pair structure | Status |
|---|---|---|---|
| purple          | WB1 | within-b `(f_{b,τ}, h_{b,τ})` — same `τ`            | already covered by `neg_xy_hat` (c1=c2 part); an explicit term double-weights the same-channel slice |
| magenta         | WB2 | within-b `(f_{b,τ}, h_{b,τ-1})` — f leads h by 1    | genuinely new |
| teal            | CB0 | cross-b  `(h_{b,τ}, f_{b',τ})` — same `τ`            | genuinely new |
| orange (bright) | CB1 | cross-b  `(f_{b,τ}, h_{b',τ})` — same `τ`            | same loss term as CB0 under `b ↔ b'`; one implementation covers both colours |
| dark red        | CB2 | cross-b  `(f_{b,τ}, h_{b',τ-1})` — f leads h by 1    | genuinely new |
| olive           | CTH | cross-b  `(h_{b,τ-1}, h_{b',τ})` — adj. `τ` (h↔h)    | genuinely new |
| brown           | CTF | cross-b  `(f_{b,τ}, f_{b',τ+1})` — adj. `τ` (f↔f)    | genuinely new |

Of the 7 candidate colours: 1 (WB1) is redundant with an existing term, and
2 (CB0 + CB1) implement the same loss term — leaving **5 distinct new
pair-types** to ablate: WB2, the merged CB0/CB1, CB2, CTH, CTF.

## Reproducing

```
python3 experiments/2026-05-09_exp_additional_negatives/scripts/plot_additional_negatives_diagram.py
```

Writes both PNGs to `plots/`.
