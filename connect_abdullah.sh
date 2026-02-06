#!/bin/bash
sshfs -o allow_other,ssh_command='ssh -4' abdullah@ml3d.vc.in.tum.de:/cluster/51/abdullah/2DGaussianSplatting .
ssh abdullah@ml3d.vc.in.tum.de  -t 'cd /cluster/51/abdullah/2DGaussianSplatting && salloc --gpus=1'
