#!/bin/bash
PREFIX="../../inputs/"
FILES=("input_100_1000x1000.txt")
PROGRAM="./acsopenmp"
THREADS=(8)
ANT="64"
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
