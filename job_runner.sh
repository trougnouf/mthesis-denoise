#!/bin/bash
while :
do
    CMD=$(sed -e 1$'{w/dev/stdout\n;d}' -i~ job_queue.sh)
    eval $CMD
    sleep 30
done

