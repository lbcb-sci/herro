1. download test data: wget -O HG002.chr19_10M_12M.fastq.gz https://zenodo.org/records/14048797/files/HG002.chr19_10M_12M.fastq.gz?download=1
2. download model: wget -O model_R10_v0.1.pt https://zenodo.org/records/12683277/files/model_v0.1.pt?download=1
3. run herro: e.g. herro inference -t 4 -m model_R10_v0.1.pt -d 2 -b 32 HG002.chr19_10M_12M.fastq.gz output.fasta


Time taken: During our test, the run took arounds 10 minutes with 4 threads, one Tesla V100-SXM2-32GB-LS and batch size 32.   
Expected output: A file in fasta format of around 80Mb.

----------------------
The data is a part of the data from 

https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/5b73fa0e-658a-4248-b2b8-cd16155bc157--UCSC_GIAB_R1041_nanopore/HG002_R1041_UL/dorado/v0.4.0_wMods/

, preprocessed with adaptor trimming/splitting and filtered to at least 10000bp and Q10,

and contains reads (~40x) that are aligned to 10-12Mbp of chr19 maternal/paternal of HG002 v1.0.1 assembly from https://github.com/marbl/HG002
