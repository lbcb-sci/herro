import argparse
import re
import sys
from contextlib import ExitStack

from Bio import SeqIO
from tqdm import tqdm


def get_cigar(cigar):
    for m in re.finditer(r'(\d+)([=XID])', cigar):
        op = m.group(2)
        if op == '=' or op == 'X':
            op = 'M'

        yield op, int(m.group(1))


def right_align_hp_indels(cigar, tseq, qseq):
    tpos, qpos = 0, 0
    to_shrink = False
    for i in range(len(cigar)):
        op, length = cigar[i]
        if length == 0:
            to_shrink = True

        if op == 'M':
            tpos += length
            qpos += length
        elif op == 'I' or op == 'D':
            if i > 0 and i < len(cigar) - 1 and cigar[
                    i - 1][0] == 'M' and cigar[i + 1][0] == 'M':
                next_len = cigar[i + 1][1]

                if op == 'I':
                    for l in range(next_len):
                        if qseq[qpos + l] != qseq[qpos + l + length]:
                            break
                elif op == 'D':
                    for l in range(next_len):
                        if tseq[tpos + l] != tseq[tpos + l + length]:
                            break
                else:
                    raise ValueError('Invalid cigar')

                if l > 0:
                    prev_op, prev_len = cigar[i - 1]
                    cigar[i - 1] = (prev_op, prev_len + l)

                    next_op, _ = cigar[i + 1]
                    cigar[i + 1] = (next_op, next_len - l)

                    tpos += l
                    qpos += l
                if l == next_len:
                    to_shrink = True

            if op == 'I':
                qpos += length
            elif op == 'D':
                tpos += length
            else:
                raise ValueError('Invalid cigar')
        else:
            raise ValueError(f'Invalid cigar:{op}')

    assert tpos == len(tseq) and qpos == len(
        qseq), 'Invalid length before and after right-aligning HP indels'

    if to_shrink:
        l = 0
        for i in range(len(cigar)):
            _, length = cigar[i]
            if length != 0:
                cigar[l] = cigar[i]
                l += 1

        cigar = cigar[:l]

    return cigar


def get_fwd_qpos(qpos, qstart, qend, strand):
    if strand == '+':
        return qpos + qstart

    return qend - 1 - qpos


def output_for_aln_type(seqs, paf_path, query_path, target_path, aln_type):
    with ExitStack() as stack:
        paf = stack.enter_context(open(paf_path, "r"))
        query = stack.enter_context(open(query_path, "w"))
        target = stack.enter_context(open(target_path, "w"))

        for line in tqdm(paf):
            line = line.split('\t')

            qname, qstart, qend = line[0], int(line[2]), int(line[3])
            qseq = seqs[qname].seq[qstart:qend]

            tname, tstart, tend = line[5], int(line[7]), int(line[8])
            tseq = str(seqs[tname].seq[tstart:tend])

            strand = line[4]
            qseq = str(
                qseq.reverse_complement()) if strand == '-' else str(qseq)

            cigar = line[-1][5:]
            cigar = list(get_cigar(cigar))
            if aln_type == 'right':
                cigar = right_align_hp_indels(cigar, tseq, qseq)

            tpos, qpos = 0, 0
            for op, length in cigar:
                if op == 'M':
                    for i in range(length):
                        try:
                            if tseq[tpos + i] != qseq[qpos + i]:
                                fwd_qpos = get_fwd_qpos(qpos + i, qstart, qend,
                                                        strand)

                                target.write(
                                    f'{tname}\t{tstart+tpos+i}\t.\tX\n')
                                query.write(f'{qname}\t{fwd_qpos}\t.\tX\n')
                        except IndexError:
                            print(tpos + i, qpos + i, len(tseq), len(qseq))

                    qpos += length
                    tpos += length
                elif op == 'D':
                    # Deletion in target -> insertion in query
                    fwd_qpos = get_fwd_qpos(qpos, qstart, qend, strand)
                    if strand == '+':
                        fwd_qpos -= 1  # left-align

                    for i in range(0, length):
                        target.write(f'{tname}\t{tstart+tpos+i}\t.\tD\n')
                    query.write(f'{qname}\t{fwd_qpos}\t{length}\tI\n')

                    tpos += length
                elif op == 'I':
                    # Insertion in target -> deletion in query
                    for i in range(0, length):
                        fwd_qpos = get_fwd_qpos(qpos + i, qstart, qend, strand)
                        query.write(f'{qname}\t{fwd_qpos}\t.\tD\n')
                    target.write(f'{tname}\t{tstart+tpos-1}\t{length}\tI\n')

                    qpos += length
                else:
                    print('Unknown CIGAR operation', op, file=sys.stderr)
                    sys.exit(1)

            assert tpos == len(tseq) and qpos == len(
                qseq
            ), f'Invalid length, {tpos}, {len(tseq)}, {qpos}, {len(qseq)}'


def main(args):
    seqs = SeqIO.to_dict(SeqIO.parse(args.seqs, 'fasta'))

    aln_type = 'left'
    query_path = f'{args.query}.left.tsv'
    target_path = f'{args.target}.left.tsv'
    output_for_aln_type(seqs, args.paf, query_path, target_path, aln_type)

    aln_type = 'right'
    query_path = f'{args.query}.right.tsv'
    target_path = f'{args.target}.right.tsv'
    output_for_aln_type(seqs, args.paf, query_path, target_path, aln_type)


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Extract informative sites from PAF file')

    parser.add_argument('seqs', type=str, help='FASTA file with sequences')
    parser.add_argument('paf', help='PAF file')
    parser.add_argument('query', help='Output query file')
    parser.add_argument('target', help='Output target file')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    main(args)