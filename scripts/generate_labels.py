import argparse
import sys
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import *

import numpy as np
import portion as P
import pysam
from Bio import SeqIO
from tqdm import tqdm


@dataclass(eq=True, frozen=True)
class InformativePosition:
    ctg: str
    pos: int
    type: str = field(hash=False, compare=False)
    length: int = 0


def parse_informative_positions(path):
    positions = set()

    with open(path, 'r') as f:
        for line in tqdm(f):
            data = line.strip().split('\t')

            pos = InformativePosition(data[0], int(data[1]), data[3],
                                      int(data[2]) if data[3] == 'I' else 0)
            if pos in positions:
                print(f'Warning: {pos} already exists!', file=sys.stderr)
            positions.add(pos)

    return positions


def parse_confident_regions(path):
    regions = set()
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split('\t')

            ctg = data[0]
            interval = P.closedopen(int(data[1]), int(data[2]))
            regions.add((ctg, interval))

    return regions


def process_clipped_bases(cigar: List[Tuple[int, int]], is_forward):
    q_start_clipped, q_end_clipped = 0, 0

    if cigar[0][0] == 4:
        _, q_start_clipped = cigar.pop(0)

    if cigar[-1][0] == 4:
        _, q_end_clipped = cigar.pop()

    if is_forward:
        return q_start_clipped, q_end_clipped
    else:
        # Swap start and end if reversed
        return q_end_clipped, q_start_clipped


def get_fwd_pos(qpos, qlen):
    return qlen - 1 - qpos


def get_ref_to_query_pairs(cigar, rpos, qpos, qlen, is_forward):
    ref_to_query = {}
    qpos_rel = qpos

    for op, l in cigar:
        if op == 0 or op == 7 or op == 8:
            for i in range(l):
                qpos_abs = qpos_rel + i if is_forward else get_fwd_pos(
                    qpos_rel + i, qlen)
                ref_to_query[(rpos + i, 0)] = (qpos_abs, 0)

            rpos += l
            qpos_rel += l
        elif op == 1:  # Insertion
            for i in range(l):
                qpos_abs = qpos_rel + i if is_forward else get_fwd_pos(
                    qpos_rel + i, qlen)
                ref_to_query[(rpos - 1, i + 1)] = (qpos_abs, 0)

            qpos_rel += l
        elif op == 2:  # Deletion
            for i in range(l):
                if is_forward:
                    ref_to_query[(rpos + i, 0)] = (qpos_rel - 1, i + 1)
                else:
                    ref_to_query[(rpos + i, 0)] = (get_fwd_pos(qpos_rel,
                                                               qlen), l - i)

            rpos += l
        else:
            raise ValueError("Invalid CIGAR operation.")

    return ref_to_query


def get_query_info_positions(ref_to_query, info_pos, is_forward, reference_end):
    query_info_pos = set()
    for info_pos in info_pos:
        if info_pos.pos >= reference_end:
            break

        if info_pos.type != 'I':
            qpos = ref_to_query[(info_pos.pos, 0)]
            query_info_pos.add(qpos[0])
        else:
            qlast = ref_to_query[(info_pos.pos, 0)]
            ilast = 0
            for i in range(1, info_pos.length + 1):
                try:
                    qlast = ref_to_query[info_pos.pos, i]
                    ilast = i

                    query_info_pos.add(qlast[0])
                except KeyError:
                    if is_forward:
                        if ilast == 0:
                            query_info_pos.add(qlast[0])
                    else:
                        if qlast[1] == 0:
                            query_info_pos.add(qlast[0] - 1)
                        else:
                            query_info_pos.add(qlast[0])
                    break

    return query_info_pos


def write_informative(info_pos, path, window_size):
    pos_for_window = defaultdict(lambda: list())
    for pos in info_pos:
        w, offset = divmod(pos, window_size)
        pos_for_window[w].append(offset)

    for w, positions in pos_for_window.items():
        positions.sort()
        positions = np.array(positions)
        np.save(path / f'{w}.info.npy', positions)


def right_align_hp_indels(cigar, tseq, qseq):
    tpos, qpos = 0, 0
    to_shrink = False
    for i in range(len(cigar)):
        op, length = cigar[i]
        if length == 0:
            to_shrink = True

        if op == 0:
            tpos += length
            qpos += length
        elif op == 1 or op == 2:
            if i > 0 and i < len(cigar) - 1 and cigar[i - 1][0] == 0 and cigar[
                    i + 1][0] == 0:
                next_len = cigar[i + 1][1]

                if op == 1:
                    for l in range(next_len):
                        if qseq[qpos + l] != qseq[qpos + l + length]:
                            break
                elif op == 2:
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

            if op == 1:
                qpos += length
            elif op == 2:
                tpos += length
            else:
                raise ValueError('Invalid cigar')
        else:
            raise ValueError('Invalid cigar')

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


def qpos_from_bam(bam_path, ref_path, info_pos_left, info_pos_right, regions,
                  window, out_path):
    out_path = Path(out_path)

    refs = SeqIO.to_dict(SeqIO.parse(ref_path, 'fasta'))
    with ExitStack() as context:
        bam = context.enter_context(pysam.AlignmentFile(bam_path, 'rb'))
        gen_info = context.enter_context(
            (out_path / 'reads_with_labels.txt').open('w'))

        record: pysam.AlignedSegment

        first_pos_left_idx = 0
        first_pos_right_idx = 0
        done = False
        for record in tqdm(bam.fetch()):
            # Considering only primary alignments
            if record.is_supplementary or record.is_secondary:
                continue

            def is_in_confident(region):
                if region[0] != record.reference_name:
                    return False
                if P.closedopen(record.reference_start,
                                record.reference_end) not in region[1]:
                    return False
                return True

            if not any(map(is_in_confident, regions)):
                # Read is not contained in any confident region
                continue

            is_forward = not record.is_reverse

            cigar = list(record.cigartuples)
            q_start_clipped, q_end_clipped = process_clipped_bases(
                cigar, is_forward)
            qlen = record.query_length
            if not is_forward:
                target = str(refs[record.reference_name].
                             seq[record.reference_start:record.reference_end])
                query = record.query_alignment_sequence

                cigar = right_align_hp_indels(cigar, target, query)

            rpos = record.reference_start
            qpos = q_start_clipped if is_forward else q_end_clipped
            ref_to_query = get_ref_to_query_pairs(cigar, rpos, qpos, qlen,
                                                  is_forward)

            # Get info pos and first_pos_idx w.r.t strand
            informative_positions = info_pos_left if is_forward else info_pos_right
            first_pos_idx = first_pos_left_idx if is_forward else first_pos_right_idx

            i = 0
            while informative_positions[first_pos_idx + i].pos < rpos:
                if first_pos_idx + i == len(informative_positions) - 1:
                    done = True
                    break
                i += 1

            if done:
                return

            # Update first_pos_idx
            first_pos_idx += i
            if is_forward:
                first_pos_left_idx = first_pos_idx
            else:
                first_pos_right_idx = first_pos_idx

            query_info_pos = get_query_info_positions(
                ref_to_query, informative_positions[first_pos_idx:], is_forward,
                record.reference_end)

            if len(query_info_pos) > 0:
                read_path: Path = out_path / record.query_name
                read_path.mkdir(exist_ok=True)

                write_informative(query_info_pos, read_path, window)
                gen_info.write(f'{record.query_name}\t{len(query_info_pos)}\n')


def main(args):
    regions = parse_confident_regions(args.regions)

    positions_left = parse_informative_positions(args.info_pos_left)
    positions_left = sorted(positions_left, key=lambda p: (p.pos, p.length))

    positions_right = parse_informative_positions(args.info_pos_right)
    positions_right = sorted(positions_right, key=lambda p: (p.pos, p.length))

    qpos_from_bam(args.bam, args.ref, positions_left, positions_right, regions,
                  args.window, args.output)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--regions', type=str, required=True)
    parser.add_argument('--info_pos_left', type=str, required=True)
    parser.add_argument('--info_pos_right', type=str, required=True)
    parser.add_argument('--bam', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
