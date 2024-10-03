#!/usr/bin/env python

import argparse
import pysam
PYSAM_MATCH=0
PYSAM_INS=1
PYSAM_DEL=2
SET_TO_EXCLUDE = {'chrEBV', 'chrM', 'M', 'GWHCBHP00000025'}


def get_sum_long_indel(record:pysam.AlignedSegment, threshold:int):
    cigar_tuples = record.cigartuples
    total_long_ins = 0
    total_long_del = 0
    for op, len in cigar_tuples:
        if op == PYSAM_INS and len > threshold:
            total_long_ins += len
        elif op == PYSAM_DEL and len > threshold:
            total_long_del += len
    return total_long_ins, total_long_del


def estimate_error_bam(bam_path, restriction_path:str, length_threshold:int):
    restriction_set = None
    if restriction_path:
        with open(restriction_path, 'r') as f:
            restriction_set = {line.rstrip() for line in f}

    error_rates = []
    mismatch_rates = []
    indel_rates = []
    num_unmapped = 0
    num_excluded = 0 
    with pysam.AlignmentFile(bam_path) as bam:
        for record in bam.fetch(until_eof=True): 
            if record.is_secondary or record.is_supplementary:
                continue
            if restriction_set and record.query_name not in restriction_set:
                continue
            if record.is_unmapped:
                num_unmapped += 1
                continue
            if record.reference_name in SET_TO_EXCLUDE:
                num_excluded += 1
                continue
            stats = record.get_cigar_stats()
            counts = stats[0] 
            num_ins = counts[1]
            num_del = counts[2]
            num_soft_clip = counts[4]
            num_hard_clip = counts[5]
            
            total_long_ins, total_long_del = get_sum_long_indel(record, length_threshold)
            
            num_M = max(counts[8] + counts[7], counts[0])   
            
            num_columns = num_M + num_ins + num_del 
            num_NM = record.get_tag('NM')
            num_mismatch = num_NM - num_ins - num_del
            
            short_ins = num_ins - total_long_ins
            short_del = num_del - total_long_del
            adjusted_num_columns = num_columns - total_long_ins - total_long_del
            
            
            error_rate = (num_mismatch + short_ins + short_del) / adjusted_num_columns * 100
            mismatch_rate = (num_mismatch) / adjusted_num_columns * 100
            indel_rate = (short_ins + short_del) /adjusted_num_columns * 100 
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
    print('num mapped and included', num_values) 
    print('num mapped and excluded', num_excluded) 
    print('num unmapped', num_unmapped) 

def parse_args():
    parser = argparse.ArgumentParser(description='Estimate the error rate in a set of reads from its alignment to a reference. Exclude indel with length > threshold.')
    parser.add_argument('-b', '--bam', type=str, help='path of the input bam/sam file.')
    parser.add_argument('-i', '--ids', type=str, help='Ignore read ids not inside the given file (one on each line).')
    parser.add_argument('-l', '--len', type=int, help='len threshold.')
    return parser.parse_args()

def main():
    args = parse_args()
    estimate_error_bam(args.bam, args.ids, args.len)
if __name__ == '__main__':
    main()
    
