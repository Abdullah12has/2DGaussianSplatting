#!/bin/bash

# SSH key path
KEY_FILE="$(dirname "$0")/key"

# Set correct permissions on key
chmod 600 "$KEY_FILE"

# SSH into server and allocate GPU
echo "Connecting to ml3d.vc.in.tum.de..."
ssh -i "$KEY_FILE" koubaa@ml3d.vc.in.tum.de -t \
    'cd /cluster/51/koubaa/abdullah/2DGaussianSplatting && salloc --gpus=1'