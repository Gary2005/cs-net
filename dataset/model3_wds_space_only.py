import io
import json
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from pathlib import Path
import zstandard as zstd
import random
dctx = zstd.ZstdDecompressor()

# =========================
# 1. 解码
# =========================
def decode_sample(sample, task):
    result = {}

    assert task in ["winrate", "duel", "nxt_kill", "nxt_death", "alive_in_the_end"], f"Unsupported task: {task}"

    for k, v in sample.items():
        if k.endswith(".npy.zst"):
            key = k[:-8]  # 去掉 ".npy.zst"

            raw = dctx.decompress(v)
            arr = np.load(io.BytesIO(raw))

            assert arr.shape[0] == 1, f"Expected 1 time step, got {arr.shape[0]}"
            arr = arr[-1]  # 取最后一个时间步的数据

            result[key] = torch.from_numpy(arr)

        elif k.endswith(".npy"):
            key = k[:-4]
            arr = np.load(io.BytesIO(v))

            # remove the time dimension for space-only model
            assert arr.shape[0] == 1, f"Expected 1 time step, got {arr.shape[0]}"
            arr = arr[-1]  # 取最后一个时间步的数据

            result[key] = torch.from_numpy(arr)

        elif k.endswith(".json.zst"):
            raw = dctx.decompress(v)
            metadata = json.loads(raw.decode("utf-8"))
            if task == "winrate":
                result["label"] = torch.tensor(metadata["winner_info"]["winner"] == "CT", dtype=torch.float32)
            elif task == "duel":
                win_duel = random.random() < 0.5
                if win_duel == True:
                    result["duel"] = torch.tensor([metadata["future_kills"][0][0], metadata["future_kills"][0][1], win_duel], dtype=torch.long)
                else:
                    result["duel"] = torch.tensor([metadata["future_kills"][0][1], metadata["future_kills"][0][0], win_duel], dtype=torch.long)
            elif task == "nxt_kill":
                if len(metadata["future_kills"]) == 0:
                    result["nxt_kill"] = torch.tensor(10, dtype=torch.long)
                else:
                    result["nxt_kill"] = torch.tensor(metadata["future_kills"][0][0], dtype=torch.long)
            elif task == "nxt_death":
                if len(metadata["future_kills"]) == 0:
                    result["nxt_death"] = torch.tensor(10, dtype=torch.long)
                else:
                    result["nxt_death"] = torch.tensor(metadata["future_kills"][0][1], dtype=torch.long)
            elif task == "alive_in_the_end":
                result["alive_in_the_end"] = torch.tensor(metadata["alive_in_the_end"], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported task: {task}")
        else:
            continue

    return result


# =========================
# 2. 自动收集 shards
# =========================
def get_shards(output_dir, split="train"):
    """
    output_dir:
        your --output-dir
    split:
        "train" or "test"
    """
    split_dir = Path(output_dir) / split

    if not split_dir.exists():
        raise ValueError(f"{split_dir} not found")

    shards = sorted(str(p) for p in split_dir.glob("shards-*.tar"))

    if len(shards) == 0:
        raise ValueError(f"No shards found in {split_dir}")

    print(f"[{split}] found {len(shards)} shards")

    return shards


# =========================
# 3. dataset
# =========================

def filter_func(sample, task):
    if task == "winrate" or task == "alive_in_the_end" or task == "nxt_kill" or task == "nxt_death":
        return True
    elif task == "duel":
        for k, v in sample.items():
            if k.endswith(".json.zst"):
                raw = dctx.decompress(v)
                metadata = json.loads(raw.decode("utf-8"))
                if len(metadata["future_kills"]) == 0:
                    return False
                return metadata["future_kills"][0][0] != 10 and metadata["future_kills"][0][1] != 10
    else:
        raise ValueError(f"Unsupported task: {task}")
            
    return False


def build_dataset(shards, task, batch_size=8):

    dataset = (
        wds.WebDataset(shards, shardshuffle=1000)
        .shuffle(1000)
        .select(lambda x: filter_func(x, task))
        .map(lambda x: decode_sample(x, task))
        .batched(batch_size, partial=True)
    )

    return dataset

def build_dataloader(output_dir, split="train", batch_size=8, num_workers=4, task=None):

    shards = get_shards(output_dir, split)

    dataset = build_dataset(shards, task, batch_size)

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

def move_to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

if __name__ == "__main__":

    test_dir = "/data/dataset"
    tasks = ["winrate", "duel", "nxt_kill", "nxt_death", "alive_in_the_end"]

    for task in tasks:
        print(f"Testing task: {task}")
        loader = build_dataloader(test_dir, split="test", batch_size=4, num_workers=0, task=task)
        for batch in loader:
            for key, value in batch.items():
                print(f"{key}: {value.shape if torch.is_tensor(value) else type(value)}")
            break