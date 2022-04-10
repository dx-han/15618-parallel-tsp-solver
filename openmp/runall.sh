#!/bin/bash
PREFIX="../inputs/"
FILES=("input_20_10x10.txt")
PROGRAM="./tspopenmp"
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
