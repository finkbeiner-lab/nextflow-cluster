docker run -itd --gpus all --mount type=bind,source="$HOME/datastudy/bin",target=/app --mount type=bind,source=/gladstone/finkbeiner,target=/gladstone/finkbeiner jdlamstein/datastudy
