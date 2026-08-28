# Research notes and epistemic boundary

This alpha deliberately distinguishes **measurement**, **model-based correction** and
**hypothesis**. A transform derived from visible page edges is model-based correction; a
pixel observed in another registered burst frame is measured evidence; a deconvolved or
inpainted single-frame pixel is only a hypothesis. Output and confidence must preserve
those distinctions.

## Decisions grounded in prior work

- Document dewarping research shows that dense backward maps are the right abstraction for
  curved pages. DewarpNet predicts 3-D shape and a backward map, while Grid Regularization
  improves learned grid consistency. This CPU alpha does not claim those learned abilities;
  it implements only a conservative dense map constrained by *observed* top/bottom page
  boundaries. See [DewarpNet (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_ICCV_2019_paper.html)
  and [Grid Regularization (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.html).
- SLIC-style superpixels motivate working with locally coherent cells rather than assuming
  independent pixels. ChromaRecover uses image-native seam-bounded polygons when present,
  retaining color/shape descriptors and measuring spatial distribution separately. See the
  [scikit-image SLIC reference and example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_mask_slic.html).
- Multi-frame mobile photography demonstrates that registration and robust aggregation can
  combine real observations across noise, exposure and motion. ChromaRecover applies the
  narrower principle to moving glare/occlusion and exposes coverage. See Google's
  [burst HDR/low-light pipeline](https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/)
  and [burst reflection removal (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/html/Prasad_Burst_Reflection_Removal_Using_Reflection_Motion_Aggregation_Cues_WACV_2023_paper.html).
- Wiener deconvolution is an inverse filter under an assumed point-spread function, not a
  guarantee of original content. The alpha generates a small bounded PSF family, rejects
  severe ringing and uses accepted results only as hypotheses. See the
  [OpenCV out-of-focus deblurring tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.html).
- Classical document recognition treats segmentation, recognition and contextual decision
  as distinct cooperating stages; shape-context work shows how coarse point distributions
  can support deformable shape matching. The alpha follows those ideas conservatively with
  downstream digit templates, projection segmentation and explicit alternatives, without
  training on user images. See [LeCun et al., document recognition (1998)](https://bottou.org/papers/lecun-98h)
  and [Belongie et al., shape contexts (2002)](https://vision.ucsd.edu/publications/2002/shape-matching-and-object-recognition-using-shape-contexts).
- The implementation uses connected-component hierarchy, contours and Hu-style topology
  primitives documented in OpenCV's
  [structural analysis and shape descriptors](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html).

## What would be needed to go further

The current image-native extractor is specifically a **bright, low-chroma seam polygon**
extractor; it is not a universal primitive detector. The next primitive milestone should
introduce one shared extractor result contract and route among complementary families:

- seam-bounded polygons for mosaics;
- circular/blob components for Ishihara-style dot plates;
- connected chromatic regions for clean graphic inputs;
- edge-bounded cells for dark-seam layouts; and
- superpixels as a conservative fallback for soft or missing cell boundaries.

Each family must be validated against matched no-structure controls before its evidence can
affect status. Multiple extractors observing the same pixels are correlated proposals, not
independent confidence votes.

Deep folds and non-developable crumples require a trained dewarping model, calibrated 3-D
capture, or multiple viewpoints. Single-image blind deblurring/inpainting can propose
readable content, but evaluation must label it as generated and compare it against hidden
ground truth. A future research track should therefore use device-held-out paired captures:
flat digital source, photographed/deformed image, burst or depth observations, and a single
source-group split to prevent leakage.
