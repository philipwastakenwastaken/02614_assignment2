#!/bin/bash
#BSUB -q hpcintro
#BSUB -J gs
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 15
#BSUB -N
#BSUB -o gs_close.out
#BSUB -u s174274@student.dtu.dk
#BSUB -B
#BSUB -N

SAVE="output.dat"
OUT="out.dat"
THREADS="1 4 8 12 16 20 24"
PARAMS="300 500 0.0 0"
EXEC="poisson_j"

lscpu

for T in ${THREADS}
do
    echo "T=${T}"
    OMP_NUM_THREADS=${T} OMP_PLACES=sockets OMP_PROC_BIND=close ${EXEC} ${PARAMS}
done