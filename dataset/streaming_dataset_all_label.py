import torch
import yaml
import random
from pathlib import Path
from torch.utils.data import IterableDataset


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

    # -------------------------
    # worker 切 shard
    # -------------------------
    def _get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return self.files

        per_worker = int(len(self.files) / worker_info.num_workers)
        worker_id = worker_info.id

        start = worker_id * per_worker
        end = start + per_worker if worker_id != worker_info.num_workers - 1 else len(self.files)

        return self.files[start:end]

    # -------------------------
    # 从 round 采样窗口
    # -------------------------
    def _sample_window(self, round_tensor, nxt_kill, nxt_death):
        # round_tensor: (num_ticks, cur_seq_len)
        num_ticks, cur_seq_len = round_tensor.shape

        # ---------- 1️⃣ 随机窗口（允许左越界） ----------
        start = random.randint(-self.ticks_per_sample + 1,
                            num_ticks - self.ticks_per_sample)

        if start >= 0:
            # 正常截取
            round_tensor = round_tensor[start:start + self.ticks_per_sample]
        elif start == -self.ticks_per_sample:
            raise ValueError("Unexpected start index: full pad not supported in DDP setting")
            # 刚好全 pad
            round_tensor = torch.full(
                (self.ticks_per_sample, cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )
        else:
            # 左侧需要 pad
            pad_left = -start
            data_len = self.ticks_per_sample - pad_left

            data_part = round_tensor[:data_len]

            left_pad = torch.full(
                (pad_left, cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )

            round_tensor = torch.cat([left_pad, data_part], dim=0)

        # ---------- 2️⃣ 处理 seq_len 维 ----------
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

        return round_tensor, nxt_kill[max(0, start + self.ticks_per_sample - 1)], nxt_death[max(0, start + self.ticks_per_sample - 1)]

    # -------------------------
    # 主迭代
    # -------------------------
    def __iter__(self):
        files = list(self._get_worker_files())

        if self.shuffle_shards:
            random.shuffle(files)

        for file in files:
            data = torch.load(file, map_location="cpu")
            round_data = data["rounds"]  # list of tensors, each tensor shape: (num_ticks, seq_len)
            targets = data["winners"]  # list of win rate labels shape: (1,)
            nxt_kill = data["nxt_kill"] # list of next kill labels shape: (num_ticks,)
            nxt_death = data["nxt_death"] # list of next death labels shape: (num_ticks,)
            alive_in_the_end = data["alive_in_the_end"] # list of alive in the end labels shape: (num_players,)

            order = list(range(len(round_data)))
            random.shuffle(order)

            for idx in order:
                round_tensor = round_data[idx]
                if targets[idx] not in ['CT', 'T']:
                    continue
                target = (targets[idx] == 'CT')
                # skip if target not in ['CT', 'T']
                clip_tensor, kill_label, death_label = self._sample_window(round_tensor, nxt_kill[idx], nxt_death[idx])
                yield clip_tensor, kill_label, death_label, target, alive_in_the_end[idx]

    def __len__(self):
        # scanning all files, so it is slow, just for reference
        total_rounds = 0
        for file in self.files:
            data = torch.load(file, map_location="cpu")
            total_rounds += len(data["rounds"])
        return total_rounds


class TickStreamingAllLabelsDataset_Duel(IterableDataset):
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

    # -------------------------
    # worker 切 shard
    # -------------------------
    def _get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return self.files

        per_worker = int(len(self.files) / worker_info.num_workers)
        worker_id = worker_info.id

        start = worker_id * per_worker
        end = start + per_worker if worker_id != worker_info.num_workers - 1 else len(self.files)

        return self.files[start:end]

    # -------------------------
    # 从 round 采样窗口
    # -------------------------
    def _sample_window(self, round_tensor, nxt_kill, nxt_death):
        # round_tensor: (num_ticks, cur_seq_len)
        num_ticks, cur_seq_len = round_tensor.shape

        # ---------- 1️⃣ 随机窗口（允许左越界） ----------
        start = random.randint(-self.ticks_per_sample + 1,
                            num_ticks - self.ticks_per_sample)

        if start >= 0:
            # 正常截取
            round_tensor = round_tensor[start:start + self.ticks_per_sample]
        elif start == -self.ticks_per_sample:
            raise ValueError("Unexpected start index: full pad not supported in DDP setting")
            # 刚好全 pad
            round_tensor = torch.full(
                (self.ticks_per_sample, cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )
        else:
            # 左侧需要 pad
            pad_left = -start
            data_len = self.ticks_per_sample - pad_left

            data_part = round_tensor[:data_len]

            left_pad = torch.full(
                (pad_left, cur_seq_len),
                self.pad_token,
                dtype=round_tensor.dtype,
            )

            round_tensor = torch.cat([left_pad, data_part], dim=0)

        # ---------- 2️⃣ 处理 seq_len 维 ----------
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

        return round_tensor, nxt_kill[max(0, start + self.ticks_per_sample - 1)], nxt_death[max(0, start + self.ticks_per_sample - 1)]

    # -------------------------
    # 主迭代
    # -------------------------
    def __iter__(self):
        files = list(self._get_worker_files())

        if self.shuffle_shards:
            random.shuffle(files)

        for file in files:
            data = torch.load(file, map_location="cpu")
            round_data = data["rounds"]  # list of tensors, each tensor shape: (num_ticks, seq_len)
            targets = data["winners"]  # list of win rate labels shape: (1,)
            nxt_kill = data["nxt_kill"] # list of next kill labels shape: (num_ticks,)
            nxt_death = data["nxt_death"] # list of next death labels shape: (num_ticks,)
            alive_in_the_end = data["alive_in_the_end"] # list of alive in the end labels shape: (num_players,)

            order = list(range(len(round_data)))
            random.shuffle(order)

            for idx in order:
                round_tensor = round_data[idx]
                if targets[idx] not in ['CT', 'T']:
                    continue


                target = (targets[idx] == 'CT')
                # skip if target not in ['CT', 'T']
                clip_tensor, kill_label, death_label = self._sample_window(round_tensor, nxt_kill[idx], nxt_death[idx])
                if kill_label == 10 or death_label == 10:
                    continue

                yield clip_tensor, kill_label, death_label, target, alive_in_the_end[idx]

    def __len__(self):
        # scanning all files, so it is slow, just for reference
        total_rounds = 0
        for file in self.files:
            data = torch.load(file, map_location="cpu")
            total_rounds += len(data["rounds"])
        return total_rounds

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = TickStreamingAllLabelsDataset("config/tfm_win_rate_fine-tuning.yaml", split="test", shuffle_shards=True)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    from tqdm import tqdm
    bar = tqdm(loader, total=len(loader))
    for batch, nxt_kill, nxt_death, target, alive_in_the_end in bar:
        print(batch.shape, nxt_kill.shape, nxt_death.shape, target.shape, alive_in_the_end.shape)
        print(batch)
        print(nxt_kill)
        print(nxt_death)
        print(target)
        print(alive_in_the_end)
        break
        
    