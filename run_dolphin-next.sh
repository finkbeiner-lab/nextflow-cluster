docker run --privileged  \
    -m 20G \
    -p 8080:80 \
    -v ~/export:/export \
     --user 35019:35001 \
    --mount type=bind,source=/gladstone/finkbeiner,target=/opt/gurobi \
    --mount type=bind,source=/home/finkbeinerlab/datastudy/bin,target=/app \
    --mount type=bind,source=/gladstone/finkbeiner/lab,target=/gladstone/finkbeiner/lab \
    --mount type=bind,source=/gladstone/finkbeiner/robodata,target=/gladstone/finkbeiner/robodata \
    --mount type=bind,source=/gladstone/finkbeiner/linsley,target=/gladstone/finkbeiner/linsley \
    --mount type=bind,source=/gladstone/finkbeiner/steve,target=/gladstone/finkbeiner/steve \
    -ti dursunturan/dolphinnext-studio \
    /bin/bash
