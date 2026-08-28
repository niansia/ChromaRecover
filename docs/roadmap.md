# Public roadmap

The roadmap is evidence-driven. It records intended direction rather than a promise of dates
or universal accuracy.

## v0.3 Public Alpha

- [x] Mask-first digital recovery with top-k support, structure and evidence artifacts.
- [x] Explicit confidence levels, abstention and retry guidance.
- [x] Experimental camera-paper, camera-screen, perspective and bounded dewarping paths.
- [x] Evidence-preserving burst fusion and optional digit-shape interpretation.
- [x] Deterministic structured and hard-negative regression gates.
- [x] Safe file, pixel, burst-count and aggregate burst budgets.
- [x] External-only SmartDoc geometry and ColorBlindnessEval semantic protocols.
- [ ] First public GitHub Actions run and GitHub security settings.
- [ ] Package-registry release after the public source release is verified.

## v0.4 priorities

- [ ] Independently licensed camera-screen and camera-paper source groups from multiple
  devices, with group-held-out evaluation.
- [ ] Additional primitive families beyond bright-seam polygon cells.
- [ ] Streaming or tiled burst fusion with a measured peak-memory contract.
- [ ] Better evidence-based automatic camera routing without status calibration leakage.
- [ ] Profile-guided progress toward the one-second 1080p target.
- [ ] More structure families and accessibility-oriented presentation presets.
- [ ] Establish a passing static type-check gate before expanding the current `py.typed`
  contract; OpenCV/NumPy array annotations need a focused cleanup rather than blanket ignores.

## Later exploration

- Optional HEIC input behind a separately audited decoder dependency.
- Webcam or web interfaces only with isolated workers and explicit privacy boundaries.
- More semantic recognizers, kept strictly downstream from recovery evidence.
- A learned surface model for deep paper deformation, only if suitable licensed training and
  held-out evaluation data become available.

New algorithms require a deterministic regression, a hard negative and documented evidence
provenance. Real captures must follow [`dataset-protocol.md`](dataset-protocol.md).
