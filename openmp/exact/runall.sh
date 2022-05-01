#!/bin/bash
PREFIX="../../inputs/"
FILES=("TSP_input_10.txt" "TSP_input_15.txt" "TSP_input_20.txt")
PROGRAM="./exactopenmp"
THREADS=(1 8 16 32 64)
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
