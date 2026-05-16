"""
CS-Net desktop launcher.

In dev mode: runs like `python demo_analysis/web_app.py`.
In frozen mode (PyInstaller): resolves bundled data paths and starts the app.
"""

import os
import sys
import threading
import webbrowser
from pathlib import Path


def _resolve_base_dir() -> Path:
    """Return the project root directory, working in both dev and frozen modes."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _get_writable_dir(name: str) -> Path:
    """Get a user-writable directory for uploads/outputs.

    In frozen mode, uses a directory next to the exe so users can find it.
    In dev mode, uses the repo's demo_analysis subdirectory.
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        writable = exe_dir / name
    else:
        writable = _resolve_base_dir() / "demo_analysis" / name
    writable.mkdir(parents=True, exist_ok=True)
    return writable


def _patch_paths() -> None:
    """Set environment variables so web_app.py and high_level_analysis.py
    can discover bundled data without touching their source."""
    base = _resolve_base_dir()
    os.environ["CSNET_ROOT_DIR"] = str(base)
    os.environ["CSNET_UPLOAD_DIR"] = str(_get_writable_dir("uploads"))
    os.environ["CSNET_OUTPUT_DIR"] = str(_get_writable_dir("outputs"))


def main() -> None:
    _patch_paths()

    # Import after path patching so module-level ROOT_DIR resolution picks up our env vars.
    from demo_analysis.web_app import app

    host = "127.0.0.1"
    port = 7860
    url = f"http://{host}:{port}"

    print(f"Starting CS-Net at {url}")
    print("Press Ctrl+C to quit.")

    # Open browser after a short delay to let Flask start.
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
