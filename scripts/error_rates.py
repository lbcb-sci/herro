#!/usr/bin/env python

import argparse
import pysam

def estimate_error_bam(bam_path, restriction_path:str):
    restriction_set = None
    if restriction_path:
        with open(restriction_path, 'r') as f:
            restriction_set = {line.rstrip() for line in f}

    error_rates = []
    mismatch_rates = []
    indel_rates = []
    num_unmapped = 0
    with pysam.AlignmentFile(bam_path) as bam:
        for record in bam.fetch(until_eof=True): # if not index, same as until_eof=True
            if record.is_secondary or record.is_supplementary:
                continue
            if restriction_set and record.query_name not in restriction_set:
                continue
            if record.is_unmapped:
                num_unmapped += 1
                continue
            stats = record.get_cigar_stats()
            counts = stats[0] 
            num_ins = counts[1]
            num_del = counts[2]
            num_soft_clip = counts[4]
            num_hard_clip = counts[5]
            
            num_M = max(counts[8] + counts[7], counts[0])   
            
            num_columns = num_M + num_ins + num_del 
            num_NM = record.get_tag('NM')
            num_mismatch = num_NM - num_ins - num_del
            
            error_rate = (num_mismatch + num_ins + num_del) / num_columns * 100
            mismatch_rate = (num_mismatch) / num_columns * 100
            indel_rate = (num_ins + num_del) /num_columns * 100 
            error_rates.append(error_rate)
            mismatch_rates.append(mismatch_rate)
            indel_rates.append(indel_rate)


    error_rates.sort()
    num_values = len(error_rates)
    median = error_rates[num_values // 2]
    print(f'mean error rate: {sum(error_rates)/num_values}%')
    print(f'mean mismatch error: {sum(mismatch_rates)/num_values}%')
    print(f'mean indel error: {sum(indel_rates)/num_values}%')
    print(f'median error rate: {median}%')
    print('num mapped', num_values) 
    print('num unmapped ', num_unmapped) 

def parse_args():
    parser = argparse.ArgumentParser(description='Estimate the error rate in a set of reads from its alignment to a reference.')
    parser.add_argument('-b', '--bam', type=str, help='path of the input bam/sam file.')
    parser.add_argument('-i', '--ids', type=str, help='Ignore read ids not inside the given file (one on each line).')
    return parser.parse_args()

def main():
    args = parse_args()
    estimate_error_bam(args.bam, args.ids)
if __name__ == '__main__':
    main()
    
