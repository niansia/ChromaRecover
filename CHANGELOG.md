# Changelog

All notable changes will be documented here. The project follows Semantic Versioning once
its public API reaches 1.0.

## Unreleased

### Added

- Add complete Traditional Chinese and Simplified Chinese README translations with a
  language switcher on every edition.
- Add a standard Apache `NOTICE` file for project copyright attribution.

### Changed

- Keep the `LICENSE` file byte-for-byte aligned with the standard Apache-2.0 text so GitHub
  can identify the repository license; move project attribution to `NOTICE`.
- Present the README architecture as a system architecture and link directly to the live
  starter issues now that the repository is public.
- Make Dependabot ignore NumPy and OpenCV major-version updates until a dedicated
  compatibility matrix validates them.

## 0.3.0a4 - 2026-08-29

### Security

- Reject burst counts outside 2–12 before decoding, preflight every frame and enforce an
  80-megapixel aggregate burst budget before allocating the fusion stack.
- Reject filesystem inputs above 100 MiB before image decoding, while retaining the measured
  50-megapixel decoded-image ceiling.
- Reject non-finite, repeated, distant, concave, undersized or singular user document corners
  before perspective processing, and make symmetric diamond ordering retain all four points.
- Ignore and report embedded ICC profiles larger than 4 MiB before LittleCMS parsing.
- Give CI and dependency-audit workflows explicit read-only repository permissions; add
  weekly Dependabot checks for Python and GitHub Actions dependencies.
- Document the untrusted-image threat model and require external process time/memory limits
  for hosted or multi-user integrations instead of claiming an unsafe thread timeout.

### Added

- Add public roadmap and security-model documents, static status badges, a concise project
  rationale, software-bug and feature-request forms, and a pull-request checklist.
- Add a repository-owned research architecture figure, README PNG and editable PowerPoint
  source showing the evidence-first pipeline and continuous red/blue validation loop.
- Add package keywords and classifiers for registry discoverability.
- State that inline typing is best-effort during Alpha until the OpenCV/NumPy static-type gate
  on the v0.4 roadmap is complete.

### Changed

- Present camera functionality as Experimental Camera Alpha rather than general research
  branding, and split lint, test-matrix and benchmark CI jobs for faster feedback.
- Advance the output schema to `0.5` for file-size provenance and aggregate burst-budget
  metadata.
- Move the high dependency endpoint to OpenCV 4.14 while retaining NumPy 1.26/OpenCV 4.10
  as the oldest CI combination.
- Remove the project-specific private blueprint extraction helper from the public surface and
  drop its otherwise-unused `pypdf` development dependency. Public external-dataset fetchers
  remain because they provide checksummed, reproducible evaluation workflows.
- Pin GitHub-maintained workflow actions to verified full release commit SHAs; Dependabot
  remains responsible for proposing audited updates.

## 0.3.0a3 - 2026-08-29

### Fixed

- Bound continuous evidence and overlays to a configurable 2048-pixel presentation space,
  while preserving full-resolution support/structure masks and recording exact artifact
  coordinate metadata. Set the measured input ceiling to 50 megapixels so common 48 MP phone
  originals remain supported without promising unsafe 120 MP operation.
- Replace the hard runner-up consensus cutoff with a continuous ambiguity penalty and add a
  nearby-scale metamorphic regression.
- Keep unaccepted digit labels in diagnostic alternatives instead of exposing a wrong guess
  as the public semantic hypothesis; document and test that `best` can be `None`.
- Permit semantics-only acceptance when three masks across at least two evidence families
  agree with adequate capture support and per-character margin; visual status and ranking
  remain unchanged. An unseen 38-image font/color group yielded 50% coverage with zero
  false acceptances.
- Improve conservative borderless-document fallback detection on low-contrast capture
  backgrounds and avoid full-frame float overlays.

### Added

- Add a reproducible, external-only SmartDoc Challenge 1 geometry evaluator and dataset
  license protocol. No third-party frames are committed or used as chromatic recovery truth.
- Add an external-only ColorBlindnessEval semantic evaluator with ground-truth digit labels,
  strict false-acceptance reporting and verified download integrity.

### Changed

- Advance the output schema to `0.4` for presentation artifact coordinate spaces.
- Bound and interleave primitive-graph proposals across evidence families to reduce latency
  without changing the generator or scorer contract.

## 0.3.0a2 - 2026-08-28

### Fixed

- Align API and project documentation with the implemented post-recovery digit semantics.
- Add an ordinary non-editable source install path and Python 3.13 to the CI support matrix.
- Include explicit `camera-paper` and `camera-screen` choices in failure reports.
- Document that the current native primitive branch is a bright-seam polygon extractor and
  define the multi-family extractor milestone without overstating present coverage.

### Security

- Remove the blueprint-embedded photograph and derived proxy mask from the distributable
  tree because independent redistribution permission has not been established. The extractor
  now produces an optional Git-ignored local research fixture only.
- Replace image-dependent regressions with deterministic generated fixtures carrying exact
  support/structure truth, and state that the golden inventory is 20 generated cases.

## 0.3.0a1 - 2026-08-28

### Fixed

- Preserve float32 image arrays across NumPy 1.x and NEP 50 scalar promotion in NumPy 2,
  preventing OpenCV Laplacian failures.
- Make `ok` use the public capture/decision confidence contract directly; mosaic evidence
  no longer substitutes a different score to cross the confidence threshold.
- Replace the Python-level quadratic deduplication loop with one preview-IoU matrix while
  preserving deterministic first-wins behavior.

### Changed

- Replace the eight-way mosaic hard gate with a continuous, auditable distribution score;
  it may qualify an evidence family but cannot raise confidence.
- Route `mode="auto"` to `camera-screen` only for a strong moire signature, with conservative
  digital fallback. Automatic paper routing remains uncalibrated and explicit.
- Reduce k-means sampling from 80k to a red-team-validated 40k while retaining two attempts.
  The faster 20k/one-attempt setting was rejected because it introduced hard-negative false
  positives.
- Test supported dependency endpoints explicitly in CI: NumPy 1.26/OpenCV 4.10 and NumPy
  2.2–2.4/OpenCV 4.13.
- Normalize repository text to LF and provide a Git-only source packaging script.

## 0.3.0a0 - 2026-08-28

### Added

- Image-native polygon primitive extraction and local population-density hypotheses for
  mosaic plates, with matched-histogram structured/negative CI benchmarks.
- Separate support masks, structure envelopes and continuous evidence maps in schema `0.2`.
- `camera-paper` and `camera-screen` profiles, observed-boundary dense dewarping, banding
  measurement, and ICC-to-sRGB conversion.
- Registered 2-12 frame burst fusion that rejects moving glare/color outliers and reports
  observed coverage and uncertainty.
- Conservative Wiener restoration hypotheses for severe blur, real debug bundles, and CLI
  access to burst/debug/dewarping controls.
- A 21-case source-grouped golden regression inventory.
- Optional post-recovery digit semantics with projection-valley segmentation, multi-font
  shape/chamfer/topology matching, alternatives, and cross-candidate acceptance rules.

### Changed

- Split confidence into structure, capture and decision levels; the compatibility
  `overall_confidence` property now aliases decision confidence.
- Reworked blur measurement at original capture scale with contrast normalization and an
  explicit low-texture indeterminate path.
- Add evidence-specific mosaic success gates while retaining hard-negative abstention.
- Advance the independently versioned result schema to `0.3` for semantic provenance.

## 0.2.0a0 - 2026-08-28

### Added

- Camera Alpha perspective/ROI normalization, photometric nuisance estimation, glare mask,
  mild moire suppression, fold/shadow warnings, and inverse mask mapping.
- Contour primitive extraction and spatial-graph grouping proposals.
- A real camera-screen regression fixture extracted from the project blueprint.

### Changed

- Retuned fine-scale structure ranking and relative coarse-component thresholds for thin,
  multi-glyph structures.
- Measure blur before analysis downsampling, accept up to 120 megapixels, and validate the
  decoded image format instead of trusting the filename suffix.
- Start the independently versioned output schema at `0.1`.

## 0.1.0a0 - 2026-08-28

### Added

- First end-to-end Chromatic Residual Ensemble baseline.
- Python `recover()` API and `chromarecover` CLI.
- Top-k binary masks, accessible overlays and versioned JSON results.
- Conservative image-quality signals, ambiguity handling and abstention.
- Deterministic synthetic structure and hard-negative generator.
- Unit, integration, metamorphic and CLI tests.
