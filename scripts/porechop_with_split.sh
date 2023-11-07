#!/bin/bash

set -e 
#set -x 

# Because porechop loads all sequences into RAM at once, we split the input file into parts
# to reduce RAM usage. 


seqkit='seqkit'
porechop='porechop'

input=$1
output_prefix=$2
split_parts=$3
num_threads=$4

format=fastq.gz
output_dir=$(dirname $output_prefix)
temp_dir="${output_dir}/split_temp"

if [ ! -d $output_dir ]; then
  mkdir $output_dir
fi

mkdir $temp_dir
$seqkit split2 $input -p $split_parts -O $temp_dir
count=1

for file in ${output_dir}/split_temp/*;
do 
    porechop_output="${output_dir}/porechopped.${count}.${format}"
    $porechop --threads $num_threads -i $file --format $format -o $porechop_output --adapter_threshold 95
    count=$((count + 1))
    rm $file
done
rm -r $temp_dir

# from https://stackoverflow.com/a/45028393
for file in ${output_dir}/porechopped.*.$format; do cat $file >> $output_prefix.${format} && rm $file || break ; done

