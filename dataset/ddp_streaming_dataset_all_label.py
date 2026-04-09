import torch
import yaml
import random
from pathlib import Path
from torch.utils.data import IterableDataset
import torch.distributed as dist


def _get_rank_info():
    """
    Get DDP rank and world size.
    Works for both single-GPU and DDP.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


class TickStreamingAllLabelsDataset(IterableDataset):
    def __init__(self, config_path, split="train", shuffle_shards=True):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg["data"]

        key = "train_data_path" if split == "train" else "val_data_path"
        self.files = [Path(f"training_data/{p}") for p in data_cfg[key]]

        self.ticks_per_sample = data_cfg["ticks_per_sample"]
        self.seq_len = data_cfg["seq_len"]
        self.pad_token = data_cfg["pad_token"]

        self.shuffle_shards = shuffle_shards

        self.split = split

    # -------------------------
    # shard 切分（DDP + worker）
    # -------------------------
    def _get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()
        rank, world_size = _get_rank_info()

        if self.split == "val":
            return self.files

        # ---------- 1️⃣ 先按 GPU(rank) 切 ----------
        files = self.files[rank::world_size]

        # ---------- 2️⃣ 再按 worker 切 ----------
        if worker_info is None:
            return files

        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        per_worker = len(files) // num_workers
        start = worker_id * per_worker
        end = start + per_worker if worker_id != num_workers - 1 else len(files)

        return files[start:end]

    # -------------------------
    # 从 round 采样窗口
    # -------------------------
    def _sample_window(self, round_tensor, nxt_kill, nxt_death):
        num_ticks, cur_seq_len = round_tensor.shape

        # ---------- 1️⃣ 随机窗口 ----------
        start = random.randint(-self.ticks_per_sample + 1,
                               num_ticks - self.ticks_per_sample)

        if start >= 0:
            round_tensor = round_tensor[start:start + self.ticks_per_sample]

        elif start == -self.ticks_per_sample:
            raise ValueError("Unexpected start index: full pad not supported in DDP setting")
            round_tensor = torch.full(
                (self.ticks_per_sample, cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )
        else:
            pad_left = -start
            data_len = self.ticks_per_sample - pad_left

            data_part = round_tensor[:data_len]

            left_pad = torch.full(
                (pad_left, cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )

            round_tensor = torch.cat([left_pad, data_part], dim=0)

        # ---------- 2️⃣ seq_len 处理 ----------
        cur_seq_len = round_tensor.shape[1]

        if cur_seq_len > self.seq_len:
            round_tensor = round_tensor[:, :self.seq_len]

        elif cur_seq_len < self.seq_len:
            seq_pad = torch.full(
                (self.ticks_per_sample, self.seq_len - cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )
            round_tensor = torch.cat([round_tensor, seq_pad], dim=1)

        # ---------- 3️⃣ label 对齐 ----------
        label_idx = max(0, start + self.ticks_per_sample - 1)

        return (
            round_tensor,
            nxt_kill[label_idx],
            nxt_death[label_idx],
        )

    # -------------------------
    # 主迭代
    # -------------------------
    def __iter__(self):
        files = list(self._get_worker_files())

        while True:  # 🔥 核心改动

            if self.shuffle_shards:
                random.shuffle(files)

            for file in files:
                data = torch.load(file, map_location="cpu")

                round_data = data["rounds"]
                targets = data["winners"]
                nxt_kill = data["nxt_kill"]
                nxt_death = data["nxt_death"]
                alive_in_the_end = data["alive_in_the_end"]

                order = list(range(len(round_data)))
                random.shuffle(order)

                for idx in order:
                    round_tensor = round_data[idx]

                    target = (targets[idx] == 'CT')

                    clip_tensor, kill_label, death_label = self._sample_window(
                        round_tensor,
                        nxt_kill[idx],
                        nxt_death[idx]
                    )

                    yield (
                        clip_tensor,
                        kill_label,
                        death_label,
                        target,
                        alive_in_the_end[idx]
                    )

    def __len__(self):
        """
        Warning: slow (loads all shards)
        """
        total_rounds = 0
        for file in self.files:
            data = torch.load(file, map_location="cpu")
            total_rounds += len(data["rounds"])
        return total_rounds


# -------------------------
# Debug usage
# -------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = TickStreamingAllLabelsDataset(
        "config/tfm_win_rate_fine-tuning.yaml",
        split="train",
        shuffle_shards=True
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    for batch, nxt_kill, nxt_death, target, alive_in_the_end in loader:
        print("batch:", batch.shape)
        print("kill:", nxt_kill.shape)
        print("death:", nxt_death.shape)
        print("target:", target.shape)
        print("alive:", alive_in_the_end.shape)
        break