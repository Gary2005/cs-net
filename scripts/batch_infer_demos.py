import argparse
import subprocess
import sys
from pathlib import Path


def iter_demo_files(input_dir: Path):
    return sorted(input_dir.glob("*.dem"))


def run_one(demo_path: Path, output_dir: Path, device: str, batch_size: int, model_root: Path):
    output_path = output_dir / (demo_path.stem + ".json")

    if output_path.exists():
        print(f"Skipping {demo_path.name} since output already exists.")
        return

    cmd = [
        sys.executable,
        "-m",
        "demo_analysis.get_round_win_rate",
        "--demo_path",
        str(demo_path),
        "--output",
        str(output_path),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--model_root",
        str(model_root),
    ]
    print(f"[run] {demo_path.name} -> {output_path}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch inference for demo files")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing .dem files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder to write .json outputs",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_root", default="cs-net-models")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_root = Path(args.model_root)

    if not input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {input_dir}")
    if not model_root.is_dir():
        raise SystemExit(f"model_root not found: {model_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    demos = iter_demo_files(input_dir)
    if not demos:
        print(f"No .dem files found under {input_dir}")
        return
    
    success = 0
    total = 0

    from tqdm import tqdm

    for demo_path in tqdm(demos):
        total += 1
        try:
            run_one(demo_path, output_dir, args.device, args.batch_size, model_root)
            success += 1
        except subprocess.CalledProcessError as e:
            print(f"Error processing {demo_path.name}: {e}")
        
    print(f"Finished processing {total} demos with {success} successes.")


if __name__ == "__main__":
    main()
