import os
import requests
from typing import List, Dict


class SeafileClient:
    def __init__(self, server: str, token: str, repo_id: str):
        self.server = server.rstrip("/")
        self.repo_id = repo_id
        self.headers = {
            "Authorization": f"Token {token}"
        }

    # -------------------------
    # 1️⃣ 列出目录内容
    # -------------------------
    def list_dir(self, path: str = "/") -> List[Dict]:
        """
        列出某个目录下的文件和文件夹
        path: Seafile 路径，如 "/", "/datasets"
        """
        url = f"{self.server}/api2/repos/{self.repo_id}/dir/"
        r = requests.get(url, headers=self.headers, params={"p": path})
        r.raise_for_status()
        return r.json()

    # -------------------------
    # 2️⃣ 下载文件
    # -------------------------
    def download_file(self, remote_path: str, local_path: str):
        """
        正确下载 Seafile 文件（两步法）
        """
        # Step 1: 获取 download link
        url = f"{self.server}/api2/repos/{self.repo_id}/file/"
        r = requests.get(
            url,
            headers=self.headers,
            params={"p": remote_path},
        )
        r.raise_for_status()

        download_url = r.text.strip('"')

        # Step 2: 真正下载文件内容
        with requests.get(download_url, stream=True) as r2:
            r2.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r2.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # -------------------------
    # 3️⃣ 上传文件
    # -------------------------
    def upload_file(self, local_path: str, remote_dir: str = "/"):
        # Step 1: 获取 upload link
        url = f"{self.server}/api2/repos/{self.repo_id}/upload-link/"
        r = requests.get(url, headers=self.headers, params={"p": remote_dir})
        r.raise_for_status()
        upload_url = r.text.strip('"')

        # Step 2: 上传
        filename = os.path.basename(local_path)
        with open(local_path, "rb") as f:
            files = {
                "file": (filename, f)
            }
            data = {
                "parent_dir": remote_dir,   # ⭐ 关键
            }
            r = requests.post(upload_url, files=files, data=data)
            r.raise_for_status()
