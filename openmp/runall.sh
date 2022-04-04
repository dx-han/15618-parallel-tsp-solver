#!/bin/bash
PREFIX="../inputs/"
FILES=("input_3_10.txt")
PROGRAM="./tspopenmp"
THREADS=(4)
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
