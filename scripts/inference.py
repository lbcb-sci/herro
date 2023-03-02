import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from subprocess import check_output
from typing import *

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

FWD_ENCODING = {ord(b): ord(b) for b in 'ACGT*.'}
FWD_ENCODING.update({ord(b): ord(b.upper()) for b in 'acgt'})
FWD_ENCODING[ord('#')] = ord('*')
FWD_ENCODING_FUNC = np.vectorize(FWD_ENCODING.__getitem__)


def parse_predictions(path: Path) -> Dict[str, Dict[int, np.array]]:
    predictions = defaultdict(lambda: dict())

    n_lines = int(check_output(["wc", "-l", f'{str(path)}']).split()[0])
    with path.open() as f:
        for line in tqdm(f, total=n_lines):
            data = line.strip().split('\t')

            read_id = data[0]
            window_id = int(data[1])
            logits = np.fromstring(data[2], dtype=np.half, sep=',')

            predictions[read_id][window_id] = logits

    return predictions


def correct_read(read_id: str, path: Path, read: Seq,
                 predictions: Dict[int, np.array], window: int) -> str:
    length = len(read)
    n_windows = (length + window - 1) // window

    corrected = ''
    for w in range(n_windows):
        feats_path = path / f'{w}.features.npy'
        if not feats_path.exists():  # No feature matrix for this window
            start, end = w * window, min((w + 1) * window, length)
            corrected += str(read[start:end]).upper()
            continue

        ids_path = path / f'{w}.ids.txt'
        with ids_path.open() as ids:
            n_ovlps = len(ids.readlines())
        if n_ovlps < 2:  # Low number of overlaps for this window
            start, end = w * window, min((w + 1) * window, length)
            corrected += str(read[start:end]).upper()
            continue

        n_rows = min(n_ovlps + 1, 31)
        features = np.load(feats_path)[:n_rows, :, 0]  # NxL
        features = FWD_ENCODING_FUNC(features)

        idx = [i for i, v in enumerate(features[0]) if v != ord('*')]
        logits = predictions[w]
        assert len(idx) == len(logits)

        for i, logit in enumerate(logits):
            if logit > 1:
                # Keep the base from the target
                corrected += chr(features[0, idx[i]])
            else:
                # Correct bases
                start = idx[i]
                end = features.shape[1] if i == len(idx) - 1 else idx[i + 1]

                for pos in range(start, end):
                    bases = Counter(features[:, pos])
                    try:
                        del bases[46]  # Remove '.'
                    except KeyError:
                        pass

                    base = chr(bases.most_common(1)[0][0])
                    if base != '*':
                        corrected += base

    return read_id, corrected


def main(args: argparse.Namespace):
    tqdm.write('Parsing reads')
    reads = SeqIO.to_dict(SeqIO.parse(args.reads, 'fastq'))

    tqdm.write('Parsing predictions')
    predictions = parse_predictions(args.predictions)

    read_folders = list(args.features.glob('*/'))
    with ProcessPoolExecutor(
            args.processes) as pool, args.output.open('w') as output:
        futures = []
        for read_path in read_folders:
            read_id = read_path.name

            future = pool.submit(correct_read, read_id, read_path,
                                 reads[read_id].seq, predictions[read_id],
                                 args.window)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            read_id, seq = future.result()

            output.write(f'>{read_id}\n')
            for i in range(0, len(seq), 80):
                output.write(f'{seq[i:i+80]}\n')


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--reads', type=Path, required=True)
    parser.add_argument('-f', '--features', type=Path, required=True)
    parser.add_argument('-p', '--predictions', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('-w', '--window', type=int, default=4096)
    parser.add_argument('--processes', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)