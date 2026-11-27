# Facet coarseness (draft for approval, not committed, not in SPEC yet)

## Baseline (200 strict, faceted-wing designs, stub brief, seed 7; before any change)
| | body (160 faceted parts) | wing (539 parts) |
|---|---|---|
| sheets per part, median (p10-p90, max) | 24 (11-37, 50) | 34 (12-66, 66) |
| distinct face directions (3 deg), median (p90) | 21 (31) | 18 (28) |
| creases under 10 deg | 29 % | 44 % |
| creases under 15 deg | 38 % | 46 % |
| crease angle, median | 24 deg | 23 deg |

## Proposed SPEC wording (§6, after the U3 edge-alignment paragraph)
- (U3b) Facet coarseness `facet_coarseness` in [0, 1] (gene, default 0.8 for faceted designs, 0 = the U3 geometry exactly). Coarseness c sets, by linear interpolation of r0/r1 constants:
  - `min_crease_deg` (15 at c = 1): no two adjacent sheets of a part meet at less than this angle unless coplanar. A shallower crease is removed by re-planning: fewer stations / wing facets, or the two sheets merged to one plane (the vertex nearest the mean plane moves; every move is a decoder note with its size).
  - panel budget per part (body 12-30, wing 4-12 per side).
  - normal budget: face normals snapped to <= 8 shared directions per side (3 deg clustering), extending edge alignment from top-view edges to face directions.
  - wings: planar facets only. Strict + c > 0 replaces continuous washout by steps (one incidence per panel, steps between panels) or none, so every strip stays a quad. The aero cost (CL alpha shift, induced drag, stall margin) is reported per design.
  - optional chine: a hard edge at the widest body station (crease >= min_crease_deg).
- Aero: the corner-drag penalty stays; report.md lists drag and clmax delta vs the smooth parent per design. No design is dropped for being coarse.

## Open points I need decided
1. `facet_coarseness` as a new gene (genome 101 -> 102, reshuffles populations again; tests pinned to frozen tables) or as a brief/CLI control applied to every faceted design? Recommend the gene, default label "high".
2. Stepped twist: lose the plain washout on strict faceted wings (aero cost reported). OK?
3. Merging a crease moves a vertex; geometry then differs from the genes (every move a note). OK as a deliberate override?
4. The 0.3 baseline and smooth designs are untouched (c only acts on faceted parts).

## Approved (user): 1 gene (genome 102, high for faceted designs by pin), 2 stepped twist, 3 vertex moves capped at 3 % of part length (assumed constant coarse_move_frac), skipped and noted over the cap. SPEC section 6 carries the final wording.
