docker run --privileged \
  --memory 20G \
  --mount type=bind,source="/home/finkbeinerlab/datastudy/bin",target=/opt/conda/envs/deep_cell_imager/bin \
  --mount type=bind,source="/gladstone/finkbeiner/lab",target=/gladstone/finkbeiner/lab \
  --mount type=bind,source="/gladstone/finkbeiner/robodata",target=/gladstone/finkbeiner/robodata \
  --mount type=bind,source="/gladstone/finkbeiner/linsley",target=/gladstone/finkbeiner/linsley \
  --mount type=bind,source="/gladstone/finkbeiner/steve",target=/gladstone/finkbeiner/steve \
  -ti vivekgr92/deep-cell-imager:latest /bin/bash

