import pandas as pd
import subprocess
import tempfile
import os
import re
from multiprocessing import Pool, cpu_count
import time
import argparse
from pathlib import Path

A_FILE = "../data/split_data/test.csv"
#train，SM, RM
B_FILE = "../data/split_data/train.csv"
OUTPUT = "../data/output/max_identity_test_vs_train.csv"
SEQ_COL = "sequence"

# Needle default parameters
GAPOPEN = 10.0
GAPEXTEND = 0.5
MATRIX = "EBLOSUM62"

# number of CPUs
N_CPU = 24

# determines how many sequences in the test.csv each worker will process, None = auto
CHUNK_SIZE = 20
B_RECORDS = None

def init_worker(b_records):
    global B_RECORDS
    B_RECORDS = b_records

def needle_identity(seq1, seq2):
    f1 = None
    f2 = None
    try:
        with tempfile.NamedTemporaryFile( mode="w",suffix=".fa",delete=False) as x:
            x.write(">seqA\n")
            x.write(seq1 + "\n")
            f1 = x.name
        with tempfile.NamedTemporaryFile( mode="w", suffix=".fa",delete=False) as y:
            y.write(">seqB\n")
            y.write(seq2 + "\n")
            f2 = y.name
        result = subprocess.run([ "needle","-asequence",f1,"-bsequence",f2, 
         "-sprotein1","-sprotein2","-sformat1", "fasta","-sformat2", "fasta",
        "-gapopen",str(GAPOPEN),"-gapextend", str(GAPEXTEND), 
        "-datafile",MATRIX, "-auto", "-stdout"],capture_output=True,text=True) 
        if result.returncode != 0:
            raise RuntimeError(
                f"Needle exit code: {result.returncode}\n"
                f"stderr:\n{result.stderr}\n"
                f"stdout:\n{result.stdout[-1000:]}")      
        match = re.search(r"#\s*Identity:\s*(\d+)\s*/\s*(\d+)",result.stdout)
        if match is None:
            raise RuntimeError("Needle completed but Identity was not found.\n" f"stdout:\n{result.stdout[-2000:]}")
        identical = int(match.group(1))
        alignment_length = int(match.group(2))
        
        if alignment_length <= 0:
            raise RuntimeError("Invalid alignment length")
        
        return identical / alignment_length * 100
    finally:
        if f1 and os.path.exists(f1):
            os.remove(f1)
        if f2 and os.path.exists(f2):
            os.remove(f2)
                        
#A chunk task
def process_A_chunk(A_chunk):
    global B_RECORDS
    results = []
    for row_idx, seqA in A_chunk:
        best_identity = None
        best_b_row = None
        best_sequence = None
        failed_count = 0
        for b_row_idx, seqB in B_RECORDS:
            try:
                identity = needle_identity(seqA, seqB)
            except Exception as e:
                failed_count += 1
                with open(f"needle_errors_{os.getpid()}.txt", "a", encoding="utf-8") as f:
                    f.write(f"A row: {row_idx + 1}\\n"
                        f"A sequence: {seqA}\\n"
                        f"B row: {b_row_idx + 1}\\n"
                        f"B sequence: {seqB}\\n"
                        f"Error: {e}\\n"
                        f"{'=' * 60}\\n")
                continue
            if identity is not None:
                if best_identity is None or identity > best_identity:
                    best_identity = identity
                    best_b_row = b_row_idx + 1
                    best_sequence = seqB
        results.append({"_row_idx": row_idx, "max_identity": best_identity, "max_match_sequence_in_B": best_sequence,"failed_alignments": failed_count,})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the maximum Needle identity for each sequence in file A with respect to file B")
    parser.add_argument("b_file", nargs="?", help="path of B.csv（also --b-file）")
    parser.add_argument("--b-file", dest="b_file_flag", help="path of B.csv")
    parser.add_argument("--output", help="output path")
    args = parser.parse_args()

    if args.b_file and args.b_file_flag:
        parser.error("Specify the file location，b_file or --b-file")

    b_file = args.b_file_flag or args.b_file or B_FILE
    output = args.output or (OUTPUT if b_file == B_FILE
        else f"max_identity_{Path(A_FILE).stem}_{Path(b_file).stem}.csv")

    start_time = time.perf_counter()
    print(f"A file: {A_FILE}")
    print(f"B file: {b_file}")
    
    A = pd.read_csv(A_FILE)
    B = pd.read_csv(b_file)
    for df in [A, B]:
        df[SEQ_COL] = (df[SEQ_COL].astype(str).str.replace(r"\s+","",regex=True).str.upper())
    A_records = list(enumerate(A[SEQ_COL]))
    B_records = list(enumerate(B[SEQ_COL]))
    print(f"A sequences: {len(A_records)}")
    print(f"B sequences: {len(B_records)}")
    print(f"Total Needle alignments: "f"{len(A_records)*len(B_records)}")
    # auto chunk
    if CHUNK_SIZE is None:
        chunk_size = max(1, len(A_records)//(N_CPU*2))
    else:
        chunk_size = CHUNK_SIZE
    A_chunks = [
        A_records[i:i+chunk_size]
        for i in range(0,len(A_records),chunk_size)]
    print(f"CPU workers: {N_CPU}")
    print(f"Chunks: {len(A_chunks)}")
    print(f"Chunk size: {chunk_size}")
    # multiprocessing
    all_results = []
    with Pool(processes=N_CPU,initializer=init_worker,initargs=(B_records,),maxtasksperchild=50) as pool:
        for i, result in enumerate(pool.imap_unordered(process_A_chunk,A_chunks),start=1):
            all_results.extend(result)
            print(f"Finished chunks "f"{i}/{len(A_chunks)}")
    # Combined results
    result_df = pd.DataFrame(all_results,columns=["_row_idx", "max_identity", "max_match_sequence_in_B", "failed_alignments",],).set_index("_row_idx")

    result_df = result_df.reindex(range(len(A)))
    A_out = A.reset_index(drop=True).join(result_df)
    A_out.to_csv(output,index=False)
    print("Output:", output)
    elapsed = time.perf_counter() - start_time
    centiseconds = round(elapsed * 100)
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    print(f"Total runtime: {hours:02d}:{minutes:02d}:"f"{rest / 100:05.2f} (HH:MM:SS)")