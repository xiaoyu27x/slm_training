import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = 8
num_proc_load_dataset = num_proc
SUBSET_SIZE = 1000000

enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

if __name__ == "__main__":
    print("=" * 70)
    print("nanoGPT OpenWebText Subset Preparation")
    print("=" * 70)
    print(f"Subset documents : {SUBSET_SIZE:,}")
    print(f"Workers          : {num_proc}")
    print("Tokenizer        : GPT-2 BPE")
    print("Validation split : 1%")
    print("=" * 70)

    dataset = load_dataset(
        "Skylion007/openwebtext",
        name="plain_text",
        num_proc=num_proc_load_dataset,
    )

    total_documents = len(dataset["train"])
    actual_subset_size = min(SUBSET_SIZE, total_documents)
    dataset["train"] = dataset["train"].select(range(actual_subset_size))

    split_dataset = dataset["train"].train_test_split(
        test_size=0.01,
        seed=2357,
        shuffle=True,
    )
    split_dataset["val"] = split_dataset.pop("test")

    print(f"Train documents : {len(split_dataset['train']):,}")
    print(f"Val documents   : {len(split_dataset['val']):,}")

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = min(1024, len(dset))
        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = (
                dset.shard(
                    num_shards=total_batches,
                    index=batch_idx,
                    contiguous=True,
                ).with_format("numpy")
            )
            arr_batch = np.concatenate(batch["ids"])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"{split}.bin finished: {idx:,} tokens")

    train_tokens = int(np.sum(tokenized["train"]["len"], dtype=np.uint64))
    val_tokens = int(np.sum(tokenized["val"]["len"], dtype=np.uint64))

    print("=" * 70)
    print("Preparation completed successfully")
    print(f"Dataset subset : {actual_subset_size:,} documents")
    print(f"Train tokens   : {train_tokens:,}")
    print(f"Val tokens     : {val_tokens:,}")
    print(f"Total tokens   : {train_tokens + val_tokens:,}")
    print("=" * 70)
