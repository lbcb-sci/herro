#!/bin/bash
set -e 
#set -x

# enter the paths to these tools, or just leave it if they are in PATH
seqkit='seqkit'
porechop='porechop'
duplex_tools='duplex_tools'

if [ "$#" -ne 3 ]; then
    echo "This script requires 3 arguments:"
    echo "1. The input sequence file. e.g. input.fastq"
    echo "2. The output prefix. e.g. output/preprocessed"
    echo "3. The number of threads to be used."
    exit 
fi


# commandline arguments
input=$1
output_prefix=$2
num_threads=$3

# format for porechop output and seqkit filtering output
format=fastq.gz


output_dir=$(dirname $output_prefix)
#input_basename=$(basename "${input%.*}")
if [ ! -d $output_dir ]; then
  mkdir $output_dir
fi

echo "The starting date/time: $(date)." 
SECONDS=0

# porechop 
porechop_output="${output_dir}/porechopped.${format}"
$porechop --threads $num_threads -i $input --format $format -o $porechop_output --adapter_threshold 95

# duplex_tools
duplex_tools_input_dir="${output_dir}/duplex_tools_input_dir"
duplex_tools_output_dir="${output_dir}/duplex_tools_output_dir"
duplex_tools_output="${duplex_tools_output_dir}/porechopped_split.fastq.gz"

mkdir $duplex_tools_input_dir
mv $porechop_output $duplex_tools_input_dir
$duplex_tools split_on_adapter --threads $num_threads --allow_multiple_splits $duplex_tools_input_dir $duplex_tools_output_dir Native

# clean up porechop
rm "${duplex_tools_input_dir}/porechopped.${format}"

# seqkit
filtered="${output_prefix}.${format}"
$seqkit seq -m 10000 -o $filtered $duplex_tools_output

# clean up duplex_tools
rm -r $duplex_tools_input_dir
rm -r $duplex_tools_output_dir

echo "The ending date/time: $(date)." 
echo "Time taken: ${SECONDS} seconds" 


