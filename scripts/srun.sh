srun -A bdes-delta-gpu --time=30:00 --nodes=1 --ntasks-per-node=16 --partition=gpuA100x4 --gpus=1 --mem=32g --pty /bin/bash
