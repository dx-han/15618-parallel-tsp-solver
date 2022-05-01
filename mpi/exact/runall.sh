#!/bin/bash
PREFIX="../../inputs/"
FILES=("input_10_100x100.txt")
PROGRAM="./exactmpi"
THREADS=(1 8)
# module load openmpi
for FILE in ${FILES[@]}
do
    for THREAD in ${THREADS[@]}
    do
        FILENAME=${PREFIX}${FILE}
        echo "================ ${FILENAME}, #Thread: ${THREAD} ================"
        mpirun -np ${THREAD} ${PROGRAM} -f ${FILENAME}
        echo
    done
done
