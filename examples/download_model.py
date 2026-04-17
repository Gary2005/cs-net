import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "gary2oos/cs-net"

FILES = {
    "alive": [
        "model2_alive_latest_model.pt",
        "model2_alive.yaml",
    ],
    "duel": [
        "model2_duel_latest_model.pt",
        "model2_duel.yaml",
    ],
    "nxt_kill": [
        "model2_kill_latest_model.pt",
        "model2_kill.yaml",
    ],
    "nxt_death": [
        "model2_death_latest_model.pt",
        "model2_death.yaml",
    ],
    "win_rate": [
        "model2_win_latest_model.pt",
        "model2_win.yaml",
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

        # delete existing folder if exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

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