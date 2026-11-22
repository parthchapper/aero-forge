# Why aero-forge keeps giving bombs with wings

## Short answer

The pipeline does not pick the most novel design. It picks the design with the least drag at dash speed. For every brief so far, each candidate fails the `power` check. The final sort is by total check violation, so it reduces to "lowest power needed at dash speed". Each extra wing, pod, joined tip or tail adds drag, so the sort removes them. What is left is the minimum that every design must have: one fat body that holds the boxes, and one small wing.

Two sizing rules make that minimum look like a bomb before the search starts:

1. The body must be at least about 1.1 m long, because of how the boxes are packed.
2. `derive` fixes the wing area at the smallest value the stall speed allows (about 0.18 m²). The 3 m span limit is only a maximum, so the wing ends up about 1 m wide.

The novelty filter cannot see any of this, because the novelty measure does not look at the body.

None of this is a coding error. Each piece comes from a logged decision. Together they produce a torpedo.

## Evidence

All numbers come from `designs/dash-500ms-3m-span/` and `designs/dash-80ms-3m-span/`.

**The winner (500 m/s run, f00):**

| part | size |
|---|---|
| body length | 1.18 m |
| body width × height | 0.22 m × 0.18 m |
| wing area (projected) | 0.181 m² |
| wing aspect ratio | 5.6 |
| wing span | about 1.0 m (brief allows 3.0 m) |

The body is longer than the wing is wide.

**Every finalist fails only `power`, and the finalists are listed in order of that failure:**

| run | f00 | f01 | f02 | f03 | f04 |
|---|---|---|---|---|---|
| 80 m/s, power margin | −3.06 | −3.28 | −3.36 | −3.45 | −3.55 |
| 500 m/s, power margin | −17867 | −19830 | −21677 | −23434 | −25018 |

The 500 m/s numbers have small `energy` and `efficiency` failures as well, but `power` is thousands of times larger. In practice the list is a drag ranking.

**All finalists have the same basic layout.** In both runs, all 20 finalists have one body and no joined tips. In the 500 m/s run, all 10 also have exactly two surfaces: one wing and one fin. The visual critic then has to choose between three almost identical torpedoes. Its own notes say so: "nearly the same plain torpedo as f02".

## The five causes, most important first

### 1. The finalists are ranked by least total violation, not by novelty

`search.py`, `select()`:

```python
ranked = sorted(_unique(good), key=lambda s: (total_violation(s.candidate), -s.candidate.style, s.index))
```

SPEC §9 (line 152) says "top 10 by novelty". In step 4, round 2, you changed this to "least total check violation", because sorting by novelty picked joined and tall-panel designs that had defects (CHANGELOG Deviations, line 258). That made sense when most finalists passed every check. With a fast dash speed, though, every candidate fails `power`. `total_violation` counts every check, including `power`, `energy` and `efficiency`, which are not on the passable bar. So the sort becomes a drag minimiser, and drag minimisers converge on bombs.

This is exactly the "most efficient design instead of a novel one" behaviour you saw.

### 2. The novelty band caps novelty at 2 × d_min

`constants.yaml`: `search_novelty_band_dmin: [1.0, 2.0]`. A finalist must be novel, but not too novel. With $d_{\min} \approx 0.11$, no finalist can be more than about 0.22 from a reference aircraft. The reference set contains `conventional` and `plank`. The 500 m/s winner sits 1.68 × d_min from `plank`. In the 80 m/s run, 6 of the 10 finalists came from the `refine` emitter, which makes small mutations ($\sigma = 0.05$) of the reference genomes. So the band admits "a slightly different normal plane" and rejects anything far from one.

### 3. The novelty measure cannot see the body

`descriptor.py`, `desc()`: the 16 numbers are the lifting-area share along $x$ (5 bins), along $|y|$ (4 bins) and along $z$ (3 bins), the vertical-area fraction, and the counts of surfaces, joined tips and bodies. Body length, body diameter and the body-to-wing size ratio are not in it. A design that is 80 % body and a sleek flying wing with a small pod can get the same descriptor. The search therefore never sees "bomb-ness" as a lack of novelty.

### 4. The body has a minimum length of about 1.1 m

`layout.py`, `size_centre_body()`: the battery, payload and avionics boxes sit in one row along the body's axis, and only the middle 38 % of the body may hold them. The nose (`body_nose_frac` = 0.22) and the tail (`body_tail_frac` = 0.40) are empty.

$$L_{\text{stack}} = 0.132 + 0.15 + 0.10 + 4(0.01) = 0.422\ \text{m}$$

$$L_{\text{body,min}} = \frac{L_{\text{stack}}}{1 - 0.22 - 0.40} = \frac{0.422}{0.38} \approx 1.11\ \text{m}$$

The fineness gene (body length ÷ diameter, 4 to 10) then sets the body's thickness. Every design with a centre body therefore starts with a body more than 1.1 m long and at least 0.18 m wide. Boxes can go into the wing only when the wing's centre section is thick enough, and a 0.18 m² wing almost never is.

### 5. The wing area is fixed at the stall minimum, and the span is far below 3 m

`derive.py`, `_requirements()`:

$$S = \frac{2\,m\,g}{\rho\,V_s^2\,C_{L,\max}} \approx 0.18\ \text{m}^2$$

Here $V_s \approx 14.9$ m/s comes from the catapult launch speed. The decoder then scales every design to this exact area (`genome.py`, `area_check_rel_tol`). `span_max_m` is only an upper limit. At the winner's aspect ratio:

$$b = \sqrt{AR \cdot S} = \sqrt{5.6 \times 0.181} \approx 1.0\ \text{m}$$

To use the full 3 m at 0.18 m², the wing would need $AR = b^2 / S \approx 50$, which is far above the gene range. So the brief says "3 m wingspan", but the pipeline gives a wing of about 1 m on a body of about 1.2 m.

### Minor: the style score rewards long, thin bodies

`style.yaml`: `body_fineness` has weight +0.126. The effect is small, but it pushes in the same direction.

## About the briefs

- **500 m/s cannot be met.** It needs about 3.9 MW (the report shows `power_required_w 3.907e+06`). `derive` does not reject it, so the search runs anyway, and the huge `power` failure dominates the ranking. The brief comment and `dash-80ms-3m-span` already say this.
- **The brief is not the root cause.** The 80 m/s run fails `power` by about 3 on every finalist and gives the same single-body layouts.
- **The 40 m/s brief has never been searched.** It is the only brief where `power` might pass. A run on it would show how the ranking behaves once the drag term drops out.

## Fixes, smallest first

| # | change | where | size | needs your approval? |
|---|---|---|---|---|
| a | Rank finalists by `bar_violation` (passable checks only), then by novelty or style, not by `total_violation`. This returns to the SPEC's intent. Keep `power`, `energy` and `efficiency` in the report, but out of the ranking. | `search.py` `select()` | 1 line | Yes. It reverses a logged step-4 decision (RULES: changes to a gate need approval). |
| b | Raise the top of the novelty band, for example to [1, 4], and let the critic's `defects` score remove the broken outliers that caused the band. | `constants.yaml` | 1 value | Yes, same decision. |
| c | Add body terms to the descriptor: body length ÷ $\sqrt{S}$ and body volume ÷ $S^{1.5}$. A bomb then reads as near the other bombs. | `descriptor.py` | ~5 lines, `DESC_SIZE` 16 → 18 | Yes. It changes an interface, and the stored history and the CVT centroids must be rebuilt. |
| d | Change the sizing: treat the stall-speed area as a minimum, not as a target. Let a gene pick a larger area or span up to `span_max_m`. Also allow boxes side by side, or in pods and wing roots, so the body is not forced past 1.1 m. | `derive.py`, `genome.py`, `layout.py` | larger | Yes. It changes interfaces and the closure. |

Fix (a) alone should stop the drag-minimising collapse. Fixes (c) and (d) attack the cause of the bomb shape itself. I suggest you do (a) and (b) first, then rerun the 80 m/s brief and compare.

## What this did not check

- The full set of evaluated designs is not saved, only the 10 finalists. So the old runs cannot be re-ranked offline to prove fix (a). A new search run is needed.
- No code was changed and no search was run.
