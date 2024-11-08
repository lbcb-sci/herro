1. download test data from https://zenodo.org/records/14048797
2. run herro, e.g. herro inference -t 4 -m model_v0.1.pt -d 2,3 -b 32 HG002.chr19_10M_12M.fastq.gz output.fasta



----------------------
The data is a part of the data from 

https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/5b73fa0e-658a-4248-b2b8-cd16155bc157--UCSC_GIAB_R1041_nanopore/HG002_R1041_UL/dorado/v0.4.0_wMods/

, preprocessed with adaptor trimming/splitting and filtered to at least 10000bp and Q10,

and contains reads (~40x) that are aligned to 10-12Mbp of chr19 maternal/paternal of HG002 v1.0.1 assembly from https://github.com/marbl/HG002
