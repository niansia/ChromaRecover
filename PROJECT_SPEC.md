# ChromaRecover Project Specification

## North star

ChromaRecover recovers candidate spatial structures whose evidence is carried mainly by
relative color differences. It returns masks, visual evidence, quality warnings and an
explicit confidence state. It does not diagnose color vision and does not promise to
reconstruct the physically true colors or original pixels from a single photograph.

## Stable alpha contract

- `recover(image, mode="digital", semantics="none", top_k=3)` is the single-image entry
  point; camera modes additionally accept `roi`, `document_corners`, `auto_rectify`, and
  `auto_dewarp`. `recover_burst(images, ...)` is the multi-observation entry point.
- A readable but inconclusive image returns a normal `RecoverResult` with status
  `uncertain`; invalid input raises a typed exception.
- Core recovery is mask-first. Optional semantic recognizers read selected evidence after
  recovery and cannot create, rerank or become the only evidence for a mask.
- `semantics="auto"` currently means digit sequences only. It runs after mask/evidence
  selection, exposes alternatives and may accept a result only through a trusted top visual
  candidate or strict cross-family shape agreement. Shape agreement never changes the visual
  candidate rank, mask, confidence or `uncertain` status.
- Results are local-only by default and contain no telemetry.
- Filesystem inputs are format-sniffed and bounded by compressed-file and decoded-pixel
  budgets. ICC metadata has an independent parser budget. Burst frame count and aggregate
  pixels are rejected before full decoding.
- User document corners must be finite, distinct, near the image, convex, non-degenerate and
  capable of producing an invertible transform; invalid geometry raises `InvalidInputError`.
- The output schema is versioned independently from the algorithm.
- `best` is `Candidate | None`; an inconclusive low-information image may contain no
  defensible candidate at all.
- Alpha confidence has three explicit levels: structure, capture and decision. The legacy
  `overall_confidence` alias means decision confidence and will not contradict status.
- `ok` must imply `decision_confidence >= config.confidence_threshold`; no evidence-family
  score may replace that confidence for status selection.
- `mode="auto"` may route a strong screen signature to `camera-screen`. Uncalibrated paper
  classification falls back to digital rather than guessing a camera profile.

## Internal evidence layers

1. Original observation: decoded RGB plus immutable input metadata.
2. Nuisance estimates: low-frequency illumination and image-quality signals. These are
   estimates, never claimed as ground truth.
3. Relative color evidence: global Lab chroma, local Lab residuals and normalized opponent
   channels.
4. Structure candidates: multiple masks from clustering and projection families.
5. Primitives and groups: connected contours, image-native polygon cells, bounded spatial
   graphs and local primitive-population distributions.
6. Structure presentation: a binary support class, continuous evidence map and spatial
   envelope are separate outputs; presentation cannot modify ranking. Support and structure
   masks retain original coordinates, while memory-bounded evidence/overlay artifacts carry
   explicit presentation-space dimensions and scale metadata.
7. Provenance: geometry maps, capture profile, invalid support and multi-frame transforms
   remain attached to results.
8. Optional semantics: projection-based glyph segmentation plus multi-font shape/topology
   matching. It reads evidence but cannot write masks.

## Non-goals for v0.3 Experimental Alpha

- Medical diagnosis, universal natural-image segmentation or physical color restoration.
- OCR-driven mask selection, cloud uploads, telemetry, a mobile app or a required GPU.
- General or open-ended semantic interpretation, learned segmentation, or an assurance that
  a single observation can restore information destroyed by glare, clipping, blur, folds or
  undersampling.
- Treating Wiener deconvolution or any future generative inpainting as measured truth.
- Full 3-D recovery of deep, self-occluding or non-developable paper crumples.
- Treating `ok` as proof of a hidden message or semantic meaning. It means a coherent
  chromatic spatial candidate satisfied the configured decision contract.
- Providing a trustworthy in-process decode timeout. Hosted or multi-user integrations must
  isolate decoding and recovery in a process with external time and memory limits.

## Change rule

Changes to the public API, status meanings or JSON schema must update this specification,
tests and changelog together. New evidence families must not increase confidence merely by
duplicating an existing correlated hypothesis.

Real calibration and evaluation follow [`docs/dataset-protocol.md`](docs/dataset-protocol.md).
Engineering proxy masks may guard candidate retention but may not be presented as independent
structure ground truth.
