import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "gary2oos/cs-net"

FILES = {
    "alive": [
        "alive_fine-tuning.pth",
        "tfm_alive_fine-tuning.yaml",
    ],
    "duel": [
        "duel_fine-tuning.pth",
        "tfm_duel_fine-tuning.yaml",
    ],
    "nxt_kill": [
        "nxt_kill_fine-tuning.pth",
        "tfm_nxt_kill_fine-tuning.yaml",
    ],
    "win_rate": [
        "win_rate_fine-tuning.pth",
        "tfm_win_rate_fine-tuning.yaml",
    ],
}

TOKENIZER_FILE = "tokenizer.yaml"
OUTPUT_DIR = "./cs-net-models"


def download_file(filename):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # download tokenizer once
    tokenizer_path = download_file(TOKENIZER_FILE)

    for folder, file_list in FILES.items():
        target_dir = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(target_dir, exist_ok=True)

        # download & copy model/config files
        for filename in file_list:
            downloaded_path = download_file(filename)
            shutil.copy(downloaded_path, os.path.join(target_dir, filename))

        # copy tokenizer.yaml
        shutil.copy(tokenizer_path, os.path.join(target_dir, TOKENIZER_FILE))

    print("Done!")


if __name__ == "__main__":
    main()