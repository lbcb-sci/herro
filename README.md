# HERRO

HERRO (Haplotype-aware ERRor cOrrection) is a highly-accurate, haplotype-aware, deep-learning tool for error correction of Nanopore ultralong (UL) reads.

## Requirements

- Linux OS (tested on RHEL 8.6 and Ubuntu 22.04)
- [Zstandard](https://facebook.github.io/zstd/)
- Python (and conda) for data preprocessing

### Compile from source

- [libtorch 2.0.*](https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.1%2Bcu117.zip)
- [rustup](https://rustup.rs/)

## Installation

0. Clone the repository
```shell
git clone https://github.com/dominikstanojevic/herro.git
cd herro
```

1. Create conda environment
```shell
conda env create --file scripts/herro-env.yml
```

2. Build ```herro``` binary (singularity or compile from source)

    a. Build singularity image (requires sudo)
    ```shell
    sudo singularity build herro.sif herro-singularity.def
    ```
    
    Run the tool (see [Usage](#usage)) with: ```singularity run --nv herro.sif inference <args>```

    b. Compile
    
    When compiling from source, ensure that libtorch and rustup are downloaded and installed.

    ```shell
    export LIBTORCH=<libtorch_path>
    export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
    RUSTFLAGS="-Ctarget-cpu=native" cargo build -q --release
    ```
    Path to the resulting binary: ```target/release/herro```

## Usage

1. Preprocess reads
```shell
scripts/preprocess.sh <input_fastq> <output_prefix> <number_of_threads> <parts_to_split_job_into>
```
Note: Porechop loads all reads into memory, so the input may need to be split into multiple parts. Set <parts_to_split_job_into> to 1 if splitting is not needed. In Dorado v0.5, adapter trimming was added, so adapter trimming and splitting using Porechop and duplex tools will probably be removed in the future.

2. minimap2 alignment and batching

Although minimap2 can be run from the ```herro``` binary (omit --read-alns or use --write-alns to store batched alignments for future use), the preferred method is to initially run minimap2 and then utilize it to generate alignment batches. These batches will be used as input for the ```herro``` binary.

```shell
scripts/create_batched_alignments.sh <output_from_reads_preprocessing> <read_ids> <num_of_threads> <directory_for_batches_of_alignments> 
```
Note: Read ids can be obtained with seqkit: ```seqkit seq -ni <reads> > <read_ids>```

3. Error-correction
```shell
herro inference --read-alns <directory_alignment_batches> -t <feat_gen_threads_per_device> -d <gpus> -m <model_path> -b <batch_size> <preprocessed_reads> <fasta_output> 
```
Note: GPUs are specified using their IDs. For example, if the value of the parameter -d is set to 0,1,3, herro will use the first, second, and fourth GPU cards. Parameter ```-t``` is given **per device** - e.g., if ```-t``` is set to ```8``` and 3 GPUs are used, herro will create 24 feature generation theads in total. Recommended batch size is 64 for GPUs with 40 GB (possibly also for 32 GB) of VRAM and 128 for GPUs with 80 GB of VRAM. 

## Acknowledgements

This work has been supported by AI Singapore 100 Experiments (100E) Programme under the project AI-driven De Novo Diploid Assembler (AISG2-100E-2021-076) in collaboration with Agency for Science, Technology and Research (A*STAR), and Oxford Nanopore Technologies plc. (ONT).