#!/bin/bash
IMAGE=jb_tf2:0.1
PORT=7717

docker run -it --rm --runtime=nvidia \
    --entrypoint=jupyter \
    --workdir=/autoencoder \
    -v $(pwd):/autoencoder \
    -p $PORT:$PORT \
    $IMAGE \
    notebook --port=$PORT --ip=0.0.0.0 --allow-root