import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "gary2oos/CS-Net-V3"

FILES = {
    "alive": [
        "latest_alive_in_the_end.pt",
    ],
    "duel": [
        "latest_duel.pt",
    ],
    "nxt_kill": [
        "latest_nxt_kill.pt",
    ],
    "nxt_death": [
        "latest_nxt_death.pt",
    ],
    "win_rate": [
        "latest_winrate.pt",
    ],
}

OUTPUT_DIR = "./cs-net-models"


def download_file(filename):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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


    print("Done!")


if __name__ == "__main__":
    main()