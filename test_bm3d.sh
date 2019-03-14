SIGMAS=(5 10 20 30 40 50 60 70 80 90 92 95 97 99)
NOISYTESTDIR="datasets/test/tmpskull"
TESTSETS=($(ls ${NOISYTESTDIR}))
for SIGMA in ${SIGMAS[@]}; do
    for TESTSET in ${TESTSETS[@]}; do
        for TESTIMG in $(ls ${NOISYTESTDIR}/${TESTSET}); do
            mkdir -p results/test2/bm3d-${SIGMA}
            bm3d ${NOISYTESTDIR}/${TESTSET}/${TESTIMG} results/test2/bm3d-${SIGMA}/${TESTIMG} ${SIGMA} color twostep
        done
    done
done
