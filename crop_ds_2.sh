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
DSDIR=datasets/dataset

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

# parse isos, make dirs
ISOS=($(ls ${DSDIR}/${FN} | grep -o 'ISOH*[0-9]*'))
for iso in "${ISOS[@]}"
do
	mkdir -p "${DSDIR}_${CS}/${FN}/${iso}"
done
# resolution
RES=($(file ${DSDIR}/${FN}/$(ls ${DSDIR}/${FN} | head -1) | grep -o -E '[0-9]{4,}x[0-9]{3,}' | grep -o -E '[0-9]+'))
# base filename
BFN=$(file ${DSDIR}/${FN}/$(ls ${DSDIR}/${FN} | head -1) | grep -o -E '[A-Z]+_([0-9]*[a-z]*[A-Z]*-*)*')
let CURX=CURY=CROPCNT=0
while (("$CURY"<${RES[1]}))
do
	for iso in "${ISOS[@]}"
	do
		CROPPATH="${DSDIR}_${CS}/${FN}/${iso}/${BFN}_${iso}_${CROPCNT}.jpg"
		if [ -f "${CROPPATH}" ]
		then
			continue
		fi
		jpegtran -crop ${CS}x${CS}+${CURX}+${CURY} -copy none -trim -optimize -outfile ${CROPPATH} ${DSDIR}/${FN}/${BFN}_${iso}.jpg
	done
	((CROPCNT++))
	((CURX+=CS))
	if ((CURX+CS>${RES[0]}))
	then
		CURX=0
		((CURY+=CS))
		if ((CURY+CS>${RES[1]}))
		then
			echo "${FN} cropped into ${#ISOS[@]}*$(((CROPCNT+1))) pieces."
			exit 0
		fi
	fi
done
