#!/bin/bash
PREFIX="../../inputs/"
FILES=("input_5_100x100.txt")
PROGRAM="./exactopenmp"
THREADS=(1 8 16 32 64 128)
for FILE in ${FILES[@]}
do
    for THREAD in ${THREADS[@]}
    do
        FILENAME=${PREFIX}${FILE}
        echo "================ ${FILENAME}, #Thread: ${THREAD} ================"
        ${PROGRAM} -f ${FILENAME} -n ${THREAD}
        echo
    done
done
