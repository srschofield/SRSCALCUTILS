#!/bin/bash
#$ -N jn                         # Job name
#$ -q A_192T_1024G.q
#$ -l h_rt=72:00:00              # Wall‐clock limit
#$ -pe ompi-local 192            # **Request 192 cores**
#$ -l vf=5500M                   # (≈5.3 GB/core)
#$ -V                            # Export env vars
#$ -cwd                          # Run from current dir
#$ -j y                          # Join stdout & stderr
#$ -o jn_output.log              # Log file
#$ -S /bin/bash                  # Shell for job script

# (No GPU needed on this CPU‐only node:)
# Remove: #$ -l lcn_gpu=1

# Load modules / conda as before
module load apps/nvhpc/24.9/cu11.8/nvhpc
module load libs/cudnn/9.5.1.17/cuda-11

source /hpc/srs/local/miniconda3/etc/profile.d/conda.sh
conda activate apollo_castep

# Diagnostics
echo "Nodes / slots: $NSLOTS"
echo "Memory per slot: $VF"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"

# (Optional) background grep for URL
(
  sleep 60
  grep http jn_output.log > jn_server.log
) &

# Launch Jupyter on that node
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --NotebookApp.allow_origin='*'
