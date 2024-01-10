#!/usr/bin/env python
from contextlib import ExitStack
import zstandard as zstd
import argparse
from tqdm import tqdm
import sys

BATCH_SIZE = 100_000


def batch_reads(input_list, batch_size=100_000):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


def create_batches(rids, alns, outdir):
    with open(rids, 'r') as f:
        rids = [l.strip() for l in f.readlines()]
        batches = list(batch_reads(rids))

    with ExitStack() as stack:
        fds = []
        for i in range(len(batches)):
            cctx = zstd.ZstdCompressor()
            fd = stack.enter_context(
                cctx.stream_writer(open(f"{outdir}/{i}.oec.zst", "wb")))

            fd.write(f"{len(batches[i])}\n".encode())
            for rid in batches[i]:
                fd.write(f"{rid}\n".encode())

            fds.append(fd)

        rids = {}
        for i, batch in enumerate(batches):
            for rid in batch:
                rids[rid] = i

        f = stack.enter_context(sys.stdin if alns == '-' else open(alns, 'r'))
        for line in tqdm(f):
            tname = line.strip().split("\t")[5]

            if (idx := rids.get(tname, None)) is not None:
                fds[idx].write(line.encode())


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("rids", type=str)
    parser.add_argument("alignments", type=str)
    parser.add_argument("outdir", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    create_batches(args.rids, args.alignments, args.outdir)


if __name__ == "__main__":
    main()
