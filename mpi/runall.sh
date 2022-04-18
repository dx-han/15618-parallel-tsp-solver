#!/bin/bash
PREFIX="../inputs/"
FILES=("input_20_100x100.txt")
PROGRAM="./exactmpi"
THREADS=(1 8 16 32 64 128)
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
