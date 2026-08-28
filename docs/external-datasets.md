# External evaluation datasets

External images are downloaded into `tmp/external-datasets/` and are never packaged or
committed automatically. A dataset being downloadable does not mean it is compatible with
the repository's Apache-2.0 distribution.

## Approved first evaluation source

### ICDAR 2015 SmartDoc Challenge 1

- Official page: <https://sites.google.com/site/icdar15smartdoc/challenge-1/dataset>
- Archive: <https://zenodo.org/records/1230218>
- License: CC BY 4.0, with citation required.
- Contents: 120 handheld 1080p document videos (about 24,000 frames), covering perspective,
  focus/motion blur, illumination changes and partial occlusion. Every frame has document
  quadrilateral ground truth.
- ChromaRecover use: geometry and capture-quality evaluation. It is **not** chromatic-target
  ground truth and must not calibrate recovery confidence.

The official 21 MB `sampleDataset.tar.gz` contains three videos and is sufficient for the
local smoke gate:

```
python scripts/fetch_smartdoc_sample.py
python benchmark/evaluate_smartdoc.py PATH/TO/sampleDataset --assert-thresholds
```

The fetcher verifies SHA-256
`90d1a64f476ffe290ebbddf4337e108d471b9c25420551223f447db972844a9a`, rejects links and
path traversal during extraction, and writes only to the Git-ignored `tmp/` tree by default.

The initial nine-frame sample gate requires at least 90% quadrilateral detection, median
quadrilateral IoU of 0.80, and normalized corner RMSE no greater than 0.04. The current
reference run detected 9/9 frames, with median IoU 0.9075 and maximum normalized corner RMSE
0.0206. These are geometry engineering results, not recovery-confidence calibration.

### ColorBlindnessEval

- Dataset card: <https://huggingface.co/datasets/MM-Hallu/ColorBlindnessEval>
- License metadata: Apache-2.0.
- Contents: 638 generated pseudoisochromatic images with exact number, font and color-profile
  labels in the current parquet shard.
- ChromaRecover use: held-out post-recovery digit-semantic accuracy and false-acceptance
  measurement. It has no pixel mask, so it cannot establish mask IoU or calibrate recovery
  confidence.

Download and evaluate a deterministic slice without adding any image to Git:

```
python scripts/fetch_colorblindness_eval.py
python benchmark/evaluate_colorblindness.py \
  tmp/external-datasets/colorblindness-eval/data-00000-of-00001.parquet \
  --offset 600 --count 38 --assert-thresholds
```

`pyarrow` is an optional benchmark-only dependency. The fetcher verifies the dataset shard's
published Git LFS SHA-256. Because the card points to an unavailable original source, keep
the images external even though the current mirror declares Apache-2.0.

Install the reader from a repository checkout with `python -m pip install ".[research]"`;
normal ChromaRecover users do not need it.

Source groups are `(font, color_config)`, never individual images. Development diagnostics
used Arial/config 1 and the DejaVuSans/config 1 red team; the untouched final group was
DejaVuSans/config 2 (offset 600, 38 images). On that group, ChromaRecover produced candidates
for 38/38, accepted 19/38 digit readings, got all 19 accepted readings exactly right, and
made zero false acceptances. This validates bounded semantic usefulness, not pixel-mask
accuracy or medical color-vision performance.

## Research-only or license-review sources

| Dataset | Useful evidence | License decision |
| --- | --- | --- |
| [RDNet recaptured-screen dataset](https://github.com/tju-chengyijia/RDNet) | Real sRGB/RAW moire pairs | CC BY-NC-SA 4.0 and academic-only. Local non-commercial evaluation only; never redistribute in the Apache release. |
| [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) | Real curved/folded document photos plus scans | The official page requests citation but does not state a redistribution license. Local evaluation only until terms are clarified. |
| [SD7K](https://github.com/CXH-Research/DocShadow-SD7K) | 7,000 paired high-resolution document-shadow samples | Repository code is MIT; the dataset license scope is not explicit enough in the README. Do not redistribute without confirmation. |
| [UVDoc](https://github.com/tanguymagne/UVDoc-Dataset) | Paper geometry, UV maps and benchmark samples | Code repository is MIT, but final images combine third-party textures/backgrounds. Local geometry evaluation only unless per-asset rights are verified. |
| [OpticQuiz procedural method](https://opticquiz.com/research/) | Procedural pseudoisochromatic plate design | Paper is CC BY 4.0. Treat the companion implementation as algorithm reference until its software/data license is verified independently. |

Traditional Ishihara scans and random web-search images are deliberately excluded: being
visible online is not redistribution permission, and clinical plate content is commonly
copyrighted. ChromaRecover should generate its own plates or accept contributor photographs
with explicit provenance instead.

## Split and reporting rules

- Split by physical source document/plate, never by frame or crop.
- Keep third-party inputs outside Git; store only aggregate metrics and source metadata.
- Do not use a nuisance dataset as hidden-structure ground truth.
- Do not tune thresholds on the held-out test groups.
- Record the dataset version, checksum, license, selected frame IDs and citation in every
  evaluation report.

These rules extend the [camera dataset protocol](dataset-protocol.md).
