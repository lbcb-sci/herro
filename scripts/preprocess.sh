#!/bin/bash
set -e 
#set -x

# enter the paths to these tools, or just leave it if they are in PATH
script_dir=$(dirname "$0")
porechop_script="${script_dir}/porechop_with_split.sh"
no_split_script="${script_dir}/no_split.sh"
#seqkit='seqkit'
duplex_tools='duplex_tools'
#>&2 echo "seqkit version: $($seqkit version)"
>&2 echo "duplex_tools version: $($duplex_tools --version)"
>&2 echo "porechop version: $(porechop --version)"

format=fastq.gz


if [ "$#" -ne 4 ]; then
    echo "Please place porechop_with_split.sh and no_split.sh in the same directory as this script." 
    echo "This script requires 4 arguments:"
    echo "1. The input sequence file. e.g. input.fastq"
    echo "2. The output prefix. e.g. 'preprocessed' or 'output_dir/preprocessed'"
    echo "3. The number of threads to be used."
    echo "4. The number of parts to split the inputs into for porechop (since RAM usage may be high)."
    exit 
fi


# commandline arguments
input=$1
output_prefix=$2
num_threads=$3
split_parts=$4

if [ "${split_parts}" -eq 1 ]; then
  $no_split_script $input $output_prefix $num_threads
  exit    
fi


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


final_output="${output_prefix}.${format}"
#$seqkit seq -m 10000 -o $final_output $duplex_tools_output
mv $duplex_tools_output $final_output

# clean up duplex_tools
rm -r $duplex_tools_input_dir
rm -r $duplex_tools_output_dir

echo "The ending date/time: $(date)." 
echo "Time taken: ${SECONDS} seconds" 



