#!/bin/bash

EXEC = "./poisson_j"
SAVE="./output_j"
THREADS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
for T in $THREADS
do  
    echo "Result for ${T} Threads:"
    SAVETMP=${SAVE}\_${T}.dat
    time OMP_NUM_THREADS=${T} 150 5000 0.01 0 ${EXEC} >> ${SAVETMP}
    sleep(3);
done