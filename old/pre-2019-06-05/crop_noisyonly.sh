#!/usr/bin/env bash
# This script crops one (FN) image into many CSxCS crops
# Start with <CROPSIZE> run
# input:
#	args = CS run
# input:
#	args = CS FN
#	datasets/dataset/{imagesets}/{dataset name}_{FN}_ISO{ISO values}.jpg
# output:
#	datasets/dataset_CS/{imagesets}/ISO{ISO values}/{dataset name}_{FN}_ISO{ISO values}_{crop number}.jpg

# args
FN=$2	# filename
CS=$1	# crop size
DSDIR="datasets/noisyonly"
DESTDIR="datasets/test/testdata_${CS}/noisyonly"

if ! [[ "$CS" =~ ^[0-9]+$ ]] || ((CS%8!=0))
then
	echo "Syntax: bash $0 [CROPSIZE] [FILENAME] or bash $0 [CROPSIZE] run"
	echo "Error: ${CS} is an invalid crop size, CS must be a multiple of 8."
	exit -1
fi

if [ "$FN" == "run" ]
then
	NTHREADS=$(grep -c ^processor /proc/cpuinfo)
	echo "Running with $NTHREADS threads..."
	ls ${DSDIR} | xargs --max-procs=${NTHREADS} -n 1 bash $0 $1
	exit
fi

echo Cropping ${FN}
# resolution
RES=($(file ${DSDIR}/${FN} | grep -o -E '[0-9]{4,}x[0-9]{3,}' | grep -o -E '[0-9]+'))
# base filename
BFN=${FN::-4}
mkdir -p "${DESTDIR}/${BFN}"
let CURX=CURY=CROPCNT=0
while (("$CURY"<${RES[1]}))
do
	CROPPATH="$DESTDIR/${BFN}/${BFN}_${CROPCNT}.jpg"
	if [ ! -f "${CROPPATH}" ]
	then
		jpegtran -crop ${CS}x${CS}+${CURX}+${CURY} -copy none -trim -optimize -outfile ${CROPPATH} ${DSDIR}/${FN}
	fi
	((CROPCNT++))
	((CURX+=CS))
	if ((CURX+CS>${RES[0]}))
	then
		CURX=0
		((CURY+=CS))
		if ((CURY+CS>${RES[1]}))
		then
			echo "${FN} cropped into $(((CROPCNT+1))) pieces."
			exit 0
		fi
	fi
done
