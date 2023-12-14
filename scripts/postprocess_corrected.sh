#!/bin/bash

# seqkit in PATH or specify path
seqkit='seqkit'
corrected_reads=$1
output=$2
chop_len=30000
keep_len=10000

if [ "$#" -ne 2 ]; then
    echo "This is the script for chopping corrected reads for hifiasm assembly." 
    echo "This script requires 2 arguments:"
    echo "1. The input sequence file. e.g. corrected.fasta"
    echo "2. The output file name. e.g. chopped.fasta"
    exit 
fi

set -e
#set -x

$seqkit sliding -s $chop_len -W $chop_len -g $corrected_reads > "${output}.temp.fasta"
$seqkit seq -m $keep_len "${output}.temp.fasta" > $output
rm "${output}.temp.fasta"

