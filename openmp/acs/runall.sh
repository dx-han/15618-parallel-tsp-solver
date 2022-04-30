#!/bin/bash
PREFIX="../../inputs/"
FILES=("input_1024_1000x1000.txt")
PROGRAM="./acsopenmp"
THREADS=(1 8 16 32 64 128)
ANT="614"
for FILE in ${FILES[@]}
do
    for THREAD in ${THREADS[@]}
    do
        FILENAME=${PREFIX}${FILE}
        echo "================ ${FILENAME}, #Thread: ${THREAD} ================"
        ${PROGRAM} -f ${FILENAME} -n ${THREAD} -a ${ANT}
        echo
    done
done
