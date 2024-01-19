#!/bin/bash
if [ "$#" -ne 4 ]; then
	echo "Please place batch.py in the same directory as this script."
	echo "This script requires 4 arguments:"
	echo "1. The path to the preprocessed reads."
	echo "2. The path to the read ids of these reads e.g. from seqkit seq -n -i."
	echo "3. The number of threads to be used."
	echo "4. The directory to output the batches of alignments."
	exit
fi


set -e
#set -x

minimap2='minimap2'
script_dir=$(dirname "$0")
batch_script="${script_dir}/batch.py"

reads=$1
rids=$2
num_threads=$3
outdir=$4

if [ ! -d $outdir ]; then
	mkdir $outdir
fi

$minimap2 -K8g -cx ava-ont -k25 -w17 -e200 -r150 -m4000 -z200 -t${num_threads} --dual=yes $reads $reads | $batch_script $rids - $outdir






