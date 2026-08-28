# Camera dataset and annotation protocol

Camera claims must be evaluated by **source group**, not by random crops or perturbations of
one image. A source group is one original target plate/document; every device, angle, glare,
fold and compression variant derived from it stays in the same split.

## Required records

Each fixture manifest records:

- source-group ID, capture device and paper/screen profile;
- original digital source when redistribution is permitted;
- camera distance, angle, lighting and whether the image belongs to a burst;
- provenance, redistribution permission and any processing performed;
- separate support and structure references, including annotation method and annotator.

An algorithm-derived color threshold is an `engineering_proxy`, not ground truth. It may
test whether candidates retain known color support, but it cannot calibrate status or score
structure quality. Structure masks must be traced independently without viewing model output;
ambiguous pixels are marked ignore/invalid rather than forced foreground/background.

## Minimum calibration set

Before calibrating Camera Alpha status, collect at least 20 source plates with paper and
screen captures across multiple devices. Target roughly ten nuisance variants per source,
then split train/calibration/test by source-group ID. The first expansion should include at
least three independent sources: a different phone/angle screen capture, a printed paper
capture, and a wrinkled or shadowed paper capture.

Report mask IoU against support ground truth, structure-envelope IoU against independently
traced structure, abstention/retry rates, and false-positive rates on matched-histogram and
no-structure controls. Threshold selection uses calibration groups only; the held-out test
groups remain untouched until release evaluation.
