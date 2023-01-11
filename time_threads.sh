#!/bin/bash

EXEC="./poisson_j"
SAVE="./output_j"
OUTPUT="timings.log"
THREADS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
for T in $THREADS
do  
    echo "Result for ${T} Threads:" >> ${OUTPUT}
    SAVETMP=${SAVE}\_${T}.dat
    { time OMP_NUM_THREADS=${T} ${EXEC} 100 10000 0.000001 0 > ${SAVETMP} ; } 2>> ${OUTPUT}
done