# Chromatic Residual Ensemble (CRE)

CRE is the initial explainable baseline. It turns one image into several complementary
color representations, produces many candidate masks, and asks whether any candidate has
stable spatial evidence beyond primitive-sized color noise.

## Why this is not a literal three-layer decomposition

A single photograph does not uniquely identify reflectance, lighting, paper deformation,
camera processing and the original digital pixels. ChromaRecover therefore keeps the
observed pixels intact and derives *candidate* nuisance and structure layers. This avoids
presenting an underdetermined decomposition as recovered truth.

## Evidence branches

- Global Lab `a/b` preserves low-frequency chromatic structure.
- Local Lab residuals subtract a broad low-frequency field to reduce shadows and local
  white-balance drift.
- Linear-RGB opponent channels capture red-green and yellow-blue relative differences.
- A hybrid branch lets clustering use several weak cues without making any one color space
  authoritative.

Each branch generates k-means masks at several `k` values and robust PCA projection masks.
Both cluster polarities can survive; foreground is not assumed to be red, small or unique.

Camera mode preserves both the raw and corrected evidence branches. The corrected branch
uses a bounded gray-world estimate and a broad Lab luminance field; the raw branch prevents
an imperfect nuisance estimate from erasing a legitimate weak color signal. Saturated,
near-achromatic glare is marked invalid and reduces support for overlapping candidates.

## Geometry, primitives and grouping

Camera mode can use explicit document corners, an explicit ROI, a conservative large-
quadrilateral detector, or a boundary-observed dense ribbon map for moderately curved
paper. Every homography or dense sampling map is retained so masks and evidence return to
original-photo coordinates. Explicit geometry always takes precedence over automatic
estimation.

Each pixel hypothesis is also decomposed into connected contour primitives. A bounded
spatial graph links nearby, baseline-compatible components and proposes coherent groups.
These derived masks receive a complexity prior: grouping is an alternative proposal, not
independent corroboration.

For polygon mosaics, the current specialized image-native branch segments the image into
bright, low-chroma-seam-bounded cells. It does not yet detect dot fields, dark seams or soft
superpixel boundaries; those require separate extractor families. Cell
Lab color, area, fill and elongation form robust primitive descriptors; spatial position is
not included in color/shape clustering. The system then measures whether one primitive
class has an unusual *local population distribution*. This separates “which tiles share a
color role” from “where that role forms a glyph” and avoids an independent-pixel model.

The output retains all three layers: a hard support class, a multi-scale continuous density
map and a thresholded spatial envelope. The envelope makes a distributed glyph visible
without pretending every pixel inside it belongs to the foreground color.

## Capture profiles, burst evidence and restoration

Paper and screen photographs use different nuisance priors. Paper mode emphasizes smooth
illumination/fold evidence and observed-boundary geometry. Screen mode tolerates stronger
refresh banding and applies a bounded chroma-preserving moire suppression. Generic camera
mode chooses a profile from measured banding and moire scores.

`recover_burst` registers 2–12 observations with feature homographies, rejects clipped,
glare and temporal color-outlier samples, and fuses remaining pixels with local sharpness
weights. Coverage and uncertainty are explicit. This can recover an area hidden in one
frame only when another frame actually observed it.

For a single severely blurred frame, bounded Gaussian-PSF Wiener inversions are allowed as
extra feature hypotheses. Ringing-heavy outputs are rejected. They can help test whether
structure is stable under plausible restoration, but are debug artifacts—not original
pixels and not independent evidence.

## Ranking and abstention

Masks are ranked by chromatic separation, broad-scale density structure, coarse spatial
coherence, weak frame focus and capped cross-family agreement. Agreement has a small weight
because Lab and opponent channels are correlated, not independent votes. Quality signals
reduce confidence but never create a mask.

An `ok` decision additionally requires a chromatic boundary and support from the local
residual branch. Smooth color fields and achromatic stripe/moire patterns can still produce
high-scoring cluster masks, so failing this evidence gate lowers decision confidence and
forces abstention while preserving the candidate for inspection.

Mosaic evidence is summarized continuously from four named groups: spatial distribution,
chromatic support, nuisance resistance and sparse-support sufficiency. Correlated fields are
combined inside their group instead of becoming eight independent Boolean votes. The score
can establish that a mosaic evidence family exists, but status still uses capture confidence
directly; it cannot promote a sub-threshold candidate to `ok`.

Close candidates that overlap strongly count as consensus about one structure. Close
non-overlapping or complementary candidates remain ambiguous and force `uncertain`.
Confidence is intentionally capped and is not described as calibrated until independent
real camera data exists. Structure confidence describes the ranked pattern, capture
confidence applies acquisition quality, and decision confidence additionally applies
ambiguity and final-status penalties.

## Optional digit semantics

When `semantics="auto"`, the selected continuous evidence maps are segmented into a
horizontal sequence by row/column projection. Over-wide runs are split at local projection
valleys. Each normalized glyph is compared with a held-out multi-font template bank using
soft correlation, symmetric chamfer distance, soft IoU and loop topology. Results expose
the top alternatives and margins.

Acceptance requires either agreement across visual candidates or a high-quality top visual
candidate with a strong shape match. Semantics is deliberately downstream: it cannot add
pixels, repair a mask, change the recovery status or turn a sparse guess into measured
evidence. The current scope is digits `0`–`9`; handwriting beyond the tested stroke families
may abstain.

## Known Experimental Alpha limitations

- A smooth chromatic gradient or structured texture can still resemble a foreground.
- Strong blur, clipping and glare destroy information in a single observation. Burst mode
  only helps when registered alternate frames contain valid samples.
- Perspective detection abstains when no confident outer quadrilateral exists; explicit
  corners or ROI are more reliable for cropped screens and borderless paper.
- Glare is masked rather than presented as known inpainted content. Fold/shadow estimation
  corrects smooth illumination; the dense ribbon map handles observable one-axis bending,
  not deep folds, self-occlusion or arbitrary 3-D crumpling.
- Moire suppression is deliberately mild to avoid erasing thin target strokes.
- The synthetic generator is for regression and cannot substitute for device-held-out
  camera evaluation.

The research rationale and primary references are recorded in
[`research-notes.md`](research-notes.md).
