import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "gary2oos/cs-net-dataset"

FILE_NAME = "test/shards-00000.tar"

OUTPUT_DIR = "./dataset"


def download_file(filename):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )


def main():
    os.makedirs(OUTPUT_DIR + "/" + FILE_NAME.split("/")[0], exist_ok=True)


    downloaded_path = download_file(FILE_NAME)
    shutil.copy(downloaded_path, os.path.join(OUTPUT_DIR, FILE_NAME))

    print("Done!")


if __name__ == "__main__":
    main()