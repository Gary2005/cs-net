"""
从 本地 zip_archives 读取 zip，
读取其中 json，
按 round 切分，
tokenize 成 (num_ticks, seq_len)，
PAD 到 max_seq_len，
保存为 torch 训练数据。

功能：
- zip 来源：zip_archives
- 不删除 zip
- 记录已处理 json 文件名，防止重复
- 支持 resume
- 处理完成自动上传到 Seafile
"""

import os
import json
import yaml
import torch
import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

from demoparser_utils.tick_tokenizer import TickTokenizer
from scraper.seafile_client import SeafileClient

MAX_ROUND_TIME = 170
MIN_ROUND_TIME = 0

# -------------------------
# 参数
# -------------------------
MAX_SEQ_LEN = 512

ZIP_DIR = Path('/Volumes/Backup Plus/zip_archives')
OUT_DIR = Path("/Volumes/Backup Plus/training_data")

PROCESSED_JSON_LOG = Path("processed_jsons.txt")

REMOTE_SAVE_DIR = "/training_data"


# -------------------------
# 初始化 Seafile
# -------------------------
try:
    client = SeafileClient(
        server=os.environ["SEAFILE_SERVER"],
        token=os.environ["SEAFILE_TOKEN"],
        repo_id=os.environ["SEAFILE_REPO_ID"],
    )
    print("✅ Seafile client initialized")
except KeyError as e:
    print(f"⚠️ Missing environment variable: {e}. Seafile upload will be disabled.")
    client = None


# -------------------------
# 工具函数
# -------------------------
def ensure_remote_dir(path: str):
    """确保远端目录存在"""
    try:
        client.list_dir(path)
    except Exception:
        url = f"{client.server}/api2/repos/{client.repo_id}/dir/"
        requests.post(
            url,
            headers=client.headers,
            data={"p": path},
        )


def remote_file_exists(filename: str) -> bool:
    """检查远端是否已有文件"""
    try:
        items = client.list_dir(REMOTE_SAVE_DIR)
        names = [item["name"] for item in items]
        return filename in names
    except Exception:
        return False


def load_processed_jsons():
    if not PROCESSED_JSON_LOG.exists():
        return set()

    with open(PROCESSED_JSON_LOG, "r") as f:
        return set(line.strip() for line in f)


def append_processed_json(json_name):
    with open(PROCESSED_JSON_LOG, "a") as f:
        f.write(json_name + "\n")


def list_local_zips():
    """列出本地 zip"""
    return sorted([p.name for p in ZIP_DIR.glob("*.zip")])


def pad_sequence(tokens, pad_token, max_len):
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens + [pad_token] * (max_len - len(tokens))


def check_steamid_consistency(json_data):
    base_ids = {}

    for tick in json_data:
        ids = [p["steamid"] for p in tick["players_info"]]
        round_id = tick["round"]

        if round_id not in base_ids:
            base_ids[round_id] = ids

        if ids != base_ids[round_id]:
            raise ValueError("steamid 顺序不一致")


def group_by_round(json_data):
    rounds = {}

    for tick in json_data:
        rounds.setdefault(tick["round"], []).append(tick)

    return rounds


def process_json_bytes(json_bytes, tokenizer, valid_maps):
    json_data = json.loads(json_bytes)

    if not json_data:
        return [], [], [], [], [], []

    map_name = json_data[0]["map_name"]

    if map_name not in valid_maps:
        return [], [], [], [], [], []

    check_steamid_consistency(json_data)

    rounds = group_by_round(json_data)

    round_tensors = []
    nxt_kill_tensors = []
    nxt_death_tensors = []
    alive_in_the_end_tensors = []

    winners = []
    reasons = []

    for ticks in rounds.values():

        tick_tokens = []
        nxt_kill = []
        nxt_death = []
        alive_in_the_end = [None for _ in ticks[0]["players_info"]]

        times = []
        for tick in ticks:
            t = tick.get("round_seconds", None)

            if t is None:
                times.append(-1)

            times.append(t)

        if len(times) == 0:
            continue

        min_time = min(times)
        max_time = max(times)

        if max_time > MAX_ROUND_TIME or min_time < MIN_ROUND_TIME:
            # 跳过异常 round
            print(
                f"⚠️ Skip round: time range ({min_time:.2f}, {max_time:.2f})"
            )
            continue

        for tick in ticks:
            tokens = tokenizer.tokenize(tick)
            tokens = pad_sequence(tokens, tokenizer.PAD, MAX_SEQ_LEN)

            tick_tokens.append(tokens)

            steamid2num = {}
            for idx, p in enumerate(tick["players_info"]):
                steamid2num[int(p["steamid"])] = idx
            fkill = 10
            fdeath = 10
            if len(tick["future_kills"]) > 0:
                fkill = steamid2num.get(int(tick["future_kills"][0]["attacker_steamid"] if tick["future_kills"][0]["attacker_steamid"] is not None else -1), 10)
                fdeath = steamid2num.get(int(tick["future_kills"][0]["victim_steamid"] if tick["future_kills"][0]["victim_steamid"] is not None else -1), 10)
            nxt_kill.append(fkill)
            nxt_death.append(fdeath)

            if alive_in_the_end[0] is None:
                alive_in_the_end = [1 for _ in ticks[0]["players_info"]]
                for idx, p in enumerate(tick["players_info"]):
                    if int(p["steamid"]) in steamid2num:
                        if p["is_alive"] == False:
                            alive_in_the_end[steamid2num[int(p["steamid"])]] = 0
                for future_kill in tick["future_kills"]:
                    victim_steamid = int(future_kill["victim_steamid"])
                    if victim_steamid in steamid2num:
                        victim_idx = steamid2num[victim_steamid]
                        alive_in_the_end[victim_idx] = 0

        round_tensors.append(torch.tensor(tick_tokens, dtype=torch.long))
        nxt_death_tensors.append(torch.tensor(nxt_death, dtype=torch.long))
        nxt_kill_tensors.append(torch.tensor(nxt_kill, dtype=torch.long))
        alive_in_the_end_tensors.append(torch.tensor(alive_in_the_end, dtype=torch.long))

        winners.append(
            ticks[0]["round_label"]["round_info"]["winner"]
        )

        reasons.append(
            ticks[0]["round_label"]["round_info"]["reason"]
        )

        

    return round_tensors, nxt_kill_tensors, nxt_death_tensors, alive_in_the_end_tensors, winners, reasons

def debug():
    with open("demoparser_utils/tokenizer.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = TickTokenizer(config)

    json_path = "test/tricked-vs-wildcard-academy-united21-league-season-42-2388747+tricked-vs-wildcard-academy-m2-overpass.json"
    with open(json_path, "r") as f:
        json_bytes = f.read().encode("utf-8")
    rounds, nxt_kill_tensors, nxt_death_tensors, alive_in_the_end_tensors, winners, reasons = process_json_bytes(
        json_bytes,
        tokenizer,
        set(config["maps"].keys()),
    )


# -------------------------
# 主程序
# -------------------------
def main():

    ZIP_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # debug()
    # quit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=0)

    args = parser.parse_args()

    ensure_remote_dir(REMOTE_SAVE_DIR)

    # tokenizer
    with open("demoparser_utils/tokenizer.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = TickTokenizer(config)

    valid_maps = set(config["maps"].keys())

    processed_jsons = load_processed_jsons()

    zip_files = list_local_zips()

    if args.resume > 0:
        zip_files = zip_files[args.resume:]

    print(f"📦 Found {len(zip_files)} local zip archives")
    print(f"🧠 Already processed {len(processed_jsons)} json files")

    total_rounds = 0

    for zip_name in tqdm(zip_files, desc="ZIPs"):

        zip_path = ZIP_DIR / zip_name

        archive_rounds = []
        archive_nxt_kill = []
        archive_nxt_death = []
        archive_alive_in_the_end = []
        archive_winners = []
        archive_reasons = []

        save_path = OUT_DIR / f"{Path(zip_name).stem}.pt"
        
        # 如果云端存在就跳过处理
        # if remote_file_exists(save_path.name):
        #     print(f"⚠️ Remote already has {save_path.name}, skipping {zip_name}")
        #     continue

        try:

            with zipfile.ZipFile(zip_path, "r") as zf:

                json_files = [f for f in zf.namelist() if f.endswith(".json")]

                # ⭐ 如果所有 json 都处理过，直接跳过
                if all(j in processed_jsons for j in json_files):
                    print(f"⏭ Skip {zip_name}, all json processed")
                    continue

                for json_name in json_files:

                    if json_name in processed_jsons:
                        continue

                    try:

                        with zf.open(json_name) as f:
                            json_bytes = f.read()

                        rounds, nxt_kill_tensors, nxt_death_tensors, alive_in_the_end_tensors, winners, reasons = process_json_bytes(
                            json_bytes,
                            tokenizer,
                            valid_maps,
                        )

                        if rounds:

                            archive_rounds.extend(rounds)
                            archive_nxt_kill.extend(nxt_kill_tensors)
                            archive_nxt_death.extend(nxt_death_tensors)
                            archive_alive_in_the_end.extend(alive_in_the_end_tensors)
                            archive_winners.extend(winners)
                            archive_reasons.extend(reasons)

                            append_processed_json(json_name)
                            processed_jsons.add(json_name)

                    except Exception as e:
                        print(f"❌ Error {json_name}: {e}")

            if archive_rounds:

                filename = f"{Path(zip_name).stem}.pt"
                save_path = OUT_DIR / filename

                print("Starting to save training data...")

                torch.save(
                    {
                        "rounds": archive_rounds,
                        "nxt_kill": archive_nxt_kill,
                        "nxt_death": archive_nxt_death,
                        "alive_in_the_end": archive_alive_in_the_end,
                        "winners": archive_winners,
                        "reasons": archive_reasons,
                        "vocab_size": tokenizer.vocab_size(),
                        "max_seq_len": MAX_SEQ_LEN,
                    },
                    save_path,
                )

                # print(f"📤 Uploading {filename}...")

                # if remote_file_exists(filename):

                #     print(
                #         f"⚠️ Remote already has {filename}, skipping upload"
                #     )

                # else:

                #     client.upload_file(
                #         str(save_path),
                #         REMOTE_SAVE_DIR,
                #     )

                #     print(f"☁️ Uploaded {filename}")

                print(f"💾 Saved locally: {filename}")

                # 删除本地 pt
                # try:
                #     os.remove(save_path)
                # except Exception as e:
                #     print(f"⚠️ Failed to delete {filename}: {e}")

                total_rounds += len(archive_rounds)

                print(f"✅ Saved {len(archive_rounds)} rounds")

            else:

                print(f"⚠️ No new rounds in {zip_name}")

        except Exception as e:

            print(f"🔥 Fatal error processing {zip_name}: {e}")

        print(f"\n🎯 Total rounds processed so far: {total_rounds}")

    print("\n🏁 All done.")


if __name__ == "__main__":
    main()