#!/bin/bash
while :
do
    git pull
    CMD=$(sed -e 1$'{w/dev/stdout\n;d}' -i~ job_queue.sh)
    eval $CMD
    sleep 60
done

