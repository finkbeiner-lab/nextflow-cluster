# Methods — Whole-well montage neurite quantification

## Overview

Per-cell neurite morphology was quantified from live high-content fluorescence
images with an automated pipeline that (i) assembles each imaging well into a
single whole-well montage, (ii) segments cell somata with a deep-learning
instance model, (iii) detects neurites with a topology-aware convolutional
neural network trained on hand-traced ground truth, (iv) assigns each neurite to
the soma it is physically connected to, and (v) computes per-cell arborization
metrics and writes them to a relational database. Operating on the whole-well
montage rather than on individual imaging tiles allows neurites that cross tile
boundaries to be traced and attributed continuously. The pipeline is implemented
as a containerized Nextflow (DSL2) workflow executed on a Slurm cluster; all
per-cell measurements are stored in a PostgreSQL database keyed to the parent
cell records.

## Image acquisition and inputs

Live cultures were imaged on a widefield high-content microscope (Molecular
Devices ImageXpress Micro) in a 4×4 array of 16 non-overlapping fields
("sites") per well, each field 2048×2048 pixels. The morphology channel was the
cytoplasmic EGFP fill (FITC); a second channel (RFP) carried the death-indicator
signal and was not used for morphology. Neurite lengths are reported in pixels;
conversion to micrometres uses the acquisition pixel size for the objective in
use (multiply pixel lengths by the µm/pixel calibration).

## Whole-well montage assembly

For each well and timepoint, the 16 raw FITC fields were stitched into a single
8192×8192-pixel montage. Because the microscope acquires contiguous,
non-overlapping fields, tiles were placed edge-to-edge in a left-to-right,
top-to-bottom ("standard" raster) 4×4 grid with zero overlap; the absence of
usable inter-tile overlap was confirmed by normalized cross-correlation seam
scans against the instrument's own montage. Three per-tile operations were
applied before stitching:

1. **Illumination flattening (optional).** A data-driven, reference-free
   rolling-background subtraction estimated the smooth low-frequency illumination
   (vignetting and background pedestal) by a grayscale morphological opening on a
   4×-downsampled copy of each tile (footprint ~128 px at full resolution),
   followed by Gaussian smoothing, upsampling, and subtraction. Because the
   opening is computed on the downsampled image and ignores small bright objects,
   somata and thin neurites are preserved while tile-to-tile vignetting is
   removed.

2. **Empty-tile guard.** Percentile-normalizing a near-signal-free field would
   amplify its noise into a spurious bright band. Each tile's dynamic range
   (99.5th minus 1st intensity percentile) was compared to the montage-wide
   median range; tiles below 15 % of that median were written as zeros rather
   than stretched.

3. **Per-tile percentile normalization.** Each remaining tile was normalized to
   [0, 1] by clipping between its 1st and 99.5th intensity percentiles. This
   removes tile-to-tile exposure differences and matches the normalization used
   when the neurite model was trained.

The stitched, per-tile-normalized image formed the input to neurite detection;
soma segmentation was run on the untreated raw tiles (below) so that
illumination flattening affected only the neurite channel.

## Soma segmentation

Somata were segmented with Cellpose-SAM, the vision-transformer generalist model
of the Cellpose framework (Stringer & Pachitariu, 2021; Pachitariu et al.,
2025), run per tile at the 2048×2048 scale at which it was tuned. Each raw tile
was percentile-clipped (1st–99.5th) to 8-bit and segmented with diameter = 25 px,
flow threshold = 0.6, and cell-probability threshold = −1.0. The model was
instantiated in single-precision (float32) rather than the default bfloat16,
which the available GPUs (NVIDIA V100, Volta architecture) lack native support
for and therefore emulate at large speed cost. Per-tile label maps were offset
so that labels are globally unique across the montage and placed into a
montage-sized soma label image.

Two post-segmentation filters removed background detections and debris:

1. **Intensity filter.** Cells whose median morphology intensity did not exceed
   the background median plus k × robust standard deviation (MAD-based;
   1.4826 × median absolute deviation, k = 2) were discarded.

2. **Size floor.** Remaining objects smaller than max(300 px, 0.6 × the
   per-image median soma area) were discarded. The adaptive floor scales with the
   typical soma size in each field while enforcing an absolute minimum.

Surviving somata were relabeled 1…N and their shape descriptors (area,
centroid, perimeter, solidity, extent, eccentricity, major/minor axis lengths)
computed by region analysis (scikit-image; van der Walt et al., 2014).

## Neurite detection

Neurites were detected with a two-dimensional U-Net (Ronneberger et al., 2015)
trained to output a neurite-probability map. To favor the connectivity of thin,
low-contrast processes — the dominant failure mode of classical ridge filters on
this substrate — the network was trained with the soft centerline-Dice
(soft-clDice) topology loss (Shit, Paetzold et al., 2021) combined with a
supervised region loss (binary cross-entropy, positive-class weight 5;
clDice computed over 10 soft-skeletonization iterations). The network used a
depth-4 encoder–decoder with 32 base channels and a single-channel input
(~1.9 M parameters). Training inputs were 1st–99.5th-percentile-normalized.

Ground truth was generated by exhaustive manual tracing of neurites in the
morphology channel using the Simple Neurite Tracer (SNT; Arshadi et al., 2021)
across balanced fields spanning both genotypes and all source experiments; the
traces were rasterized to binary masks (line dilation radius 2 px). The network
was trained for 200 epochs (200 iterations/epoch, batch size 8, 256×256
crops, Adam, learning rate 1e-3), and the best checkpoint was selected on
held-out fields.

At inference, the trained model was applied to the full 8192×8192 montage by
sliding-window tiling (interior tile 1024 px, 64-px reflect-padded halo) so that
GPU memory stays bounded and every output pixel is written exactly once. The
probability map was thresholded at 0.5. Because per-tile normalization leaves a
straight intensity step at each interior seam that a ridge-sensitive network can
fire on, neurite detections were suppressed within a 6-px band at each interior
tile boundary; a genuine neurite crossing a seam loses only this narrow band
(negligible for length). Detected objects smaller than 25 px were removed.

## Soma-rooted per-cell attribution

Each neurite pixel was assigned to the soma it is physically connected to across
the whole montage, rather than to the nearest centroid. Ownership was computed by
a marker-controlled watershed (Beucher & Meyer, 1993) on a flat landscape seeded
by the soma labels and constrained to the foreground (the union of the neurite
mask and the somata); this propagates each soma's label outward along its
connected processes and leaves unconnected debris unassigned. The neurite mask
was skeletonized (Lee et al., 1994) and soma interiors removed to yield the
per-cell neurite skeleton.

## Per-cell neurite metrics

For every soma, the following were computed on its owned skeleton (analyzed
within the soma-plus-neurites bounding box for efficiency, with identical results
to whole-montage analysis):

- **Total neurite length** — geometric skeleton length, summing orthogonal steps
  as 1 and diagonal steps as √2 (skan; Nunez-Iglesias et al., 2018).
- **Branch points** — skeleton nodes of degree ≥ 3.
- **End points** — skeleton nodes of degree 1.
- **Primary neurites** — the number of distinct branches emerging from the
  dilated soma perimeter.
- **Maximum branch length** — the longest geodesic path along the skeleton from
  the soma, by breadth-first search restricted to the skeleton.
- **Skeleton pixels** — the attributed skeleton pixel count.

## Database integration and reproducibility

Each segmented soma was written to the `celldata` table (with a montage cell
identifier and the shape descriptors above), and its neurite metrics to the
`neuritecelldata` table, keyed by cell, tile, and channel. Writes are
idempotent: re-running a well replaces only that well's montage-level cell and
neurite records (identified by the montage cell identifier) in a single
transactional delete, leaving any independently generated per-tile cell records
untouched. This lets the analysis be re-run or extended without duplication and
integrates the per-cell neurite readouts with the existing cell, intensity, and
tracking tables of the imaging database.

## Pipeline implementation

The workflow is a Nextflow DSL2 process executed on a Slurm-scheduled compute
cluster inside a Singularity/Apptainer container; the neurite-detection and
soma-segmentation steps run on a GPU. All image processing used NumPy, SciPy, and
scikit-image; deep-learning inference used PyTorch. Model weights, tuning
parameters, and the montage geometry are set through the workflow configuration,
so the same container reproduces the analysis across experiments.

## Validation

The neurite detector was benchmarked against held-out manual SNT tracings using
pixelwise precision/recall/F1 at a 3-px tolerance. The trained clDice model
reached F1 = 0.645, versus F1 = 0.135 for a classical multiscale Frangi
vesselness filter on the identical held-out fields — an ~4.8-fold improvement,
and a decisive clearing of the ~0.25 F1 ceiling at which classical ridge filters
plateau on this low-contrast live substrate. In a head-to-head comparison against
an independent CellProfiler pipeline run on the same wells, per-well
within-cell-line reproducibility (coefficient of variation) was ~5-fold tighter
for this pipeline (median CV ≈ 0.07 vs ≈ 0.35), and the two tools were less
prone to montage illumination artifacts here because of the per-tile
normalization and flattening described above.

## Key references

All citations below were verified against primary sources (publisher records,
PubMed, and the conference proceedings); identifiers are included.

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical
  Image Segmentation. *MICCAI* 2015. LNCS 9351:234–241.
  doi:10.1007/978-3-319-24574-4_28.
- Shit S, Paetzold JC, Sekuboyina A, Ezhov I, Unger A, Zhylka A, Pluim JPW,
  Bauer U, Menze BH. clDice — a Novel Topology-Preserving Loss Function for
  Tubular Structure Segmentation. *CVPR* 2021:16560–16569.
- Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm
  for cellular segmentation. *Nature Methods* 2021;18:100–106.
  doi:10.1038/s41592-020-01018-x.
- Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization
  for cellular segmentation. *bioRxiv* 2025 (preprint).
  doi:10.1101/2025.04.28.651001.
- Nunez-Iglesias J, Blanch AJ, Looker O, Dixon MW, Tilley L. A new Python library
  to analyse skeleton images confirms malaria parasite remodelling of the red
  blood cell membrane skeleton (skan). *PeerJ* 2018;6:e4312.
  doi:10.7717/peerj.4312.
- Beucher S, Meyer F. The morphological approach to segmentation: the watershed
  transformation. In: Dougherty ER, ed. *Mathematical Morphology in Image
  Processing*. Marcel Dekker; 1993:433–481.
- Lee TC, Kashyap RL, Chu CN. Building skeleton models via 3-D medial surface/axis
  thinning algorithms. *CVGIP: Graphical Models and Image Processing*
  1994;56(6):462–478. doi:10.1006/cgip.1994.1042.
- Frangi AF, Niessen WJ, Vincken KL, Viergever MA. Multiscale vessel enhancement
  filtering. *MICCAI* 1998. LNCS 1496:130–137. doi:10.1007/BFb0056195.
  (classical baseline)
- Arshadi C, Günther U, Eddison M, Harrington KIS, Ferreira TA. SNT: a unifying
  toolbox for quantification of neuronal anatomy. *Nature Methods*
  2021;18(4):374–377. doi:10.1038/s41592-021-01105-7.
- van der Walt S, Schönberger JL, Nunez-Iglesias J, Boulogne F, Warner JD,
  Yager N, Gouillart E, Yu T. scikit-image: image processing in Python. *PeerJ*
  2014;2:e453. doi:10.7717/peerj.453.
