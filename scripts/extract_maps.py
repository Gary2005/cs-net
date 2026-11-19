from pathlib import Path
import zipfile

directory = Path(__file__).resolve().parent.parent / "maps" / "optimized_obj_files"

for zip_path in directory.glob("*.obj.zip"):
    obj_path = directory / zip_path.name.removesuffix(".zip")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(directory)

    print(f"Extracted: {obj_path.name}")

print("Map extraction completed.")
