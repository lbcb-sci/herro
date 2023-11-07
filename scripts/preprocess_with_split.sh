#!/bin/bash
set -e 
#set -x


script_dir=$(dirname "$0")
porechop_script="${script_dir}/porechop_with_split.sh"
seqkit='seqkit'
duplex_tools='duplex_tools'

format=fastq.gz

# commandline arguments
input=$1
output_prefix=$2
num_threads=$3
split_parts=$4

output_dir=$(dirname $output_prefix)

if [ ! -d $output_dir ]; then
  mkdir $output_dir
fi

echo "The starting date/time: $(date)." 
SECONDS=0

# porechop
porechop_output="${output_dir}/porechopped.${format}"
$porechop_script $input "${output_dir}/porechopped" $split_parts $num_threads

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



