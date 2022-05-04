#!/bin/bash
PREFIX="../../inputs/"
FILES=("input_100_1000x1000.txt")
PROGRAM="./acsmpi"
THREADS=(1 4 8)
ANT="60"
# module load openmpi
for FILE in ${FILES[@]}
do
    for THREAD in ${THREADS[@]}
    do
        FILENAME=${PREFIX}${FILE}
        echo "================ ${FILENAME}, #Thread: ${THREAD} ================"
        mpirun -np ${THREAD} ${PROGRAM} -f ${FILENAME} -a ${ANT}
        echo
    done
done
