"""
Reconstruct point txt files: read index,label,x,y,z; write x y z r g b (red) with space delimiter.
Original files unchanged. New files named {ID}_Seat.txt etc. Skip when new file already exists.
"""
import os
import re

# Source filename -> destination suffix (dest = {normalized_id}{suffix})
SOURCE_TO_DEST = {
    "Seat_Points.txt": "_Seat.txt",
    "SeatCushion_Points.txt": "_SeatCushion.txt",
    "FrontWheel_Points.txt": "_FrontWheel.txt",
    "RearWheel_Points.txt": "_RearWheel.txt",
}
RED_RGB = (255, 0, 0)


def _normalize_id(dir_name: str) -> str:
    s = dir_name.replace("_", "")
    match = re.match(r"^([A-Za-z]+\d+).*", s)
    return match.group(1) if match else s


def _find_file_in_dir(root_path: str, filename: str) -> str | None:
    for dirpath, _dirs, files in os.walk(root_path):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def _parse_rows_to_xyz(path: str) -> list[tuple[float, float, float]] | None:
    """Parse file: each row index, label, x, y, z (whitespace delimited). Return list of (x,y,z) in order, or None on parse error."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                return None
            rows.append((x, y, z))
    return rows


def _write_xyz_rgb(path: str, xyz_rows: list[tuple[float, float, float]], r: int, g: int, b: int) -> None:
    with open(path, "w") as f:
        for x, y, z in xyz_rows:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def run_reconstruct(dir_path: str = "digitizations") -> None:
    """
    For each ID-level directory under dir_path, reconstruct the four point file types into
    {ID}_Seat.txt, {ID}_SeatCushion.txt, {ID}_FrontWheel.txt, {ID}_RearWheel.txt.
    Each output row is x y z r g b (space-delimited, red color). Skips when the output file already exists.
    """
    if not os.path.isdir(dir_path):
        return
    for name in os.listdir(dir_path):
        level1 = os.path.join(dir_path, name)
        if not os.path.isdir(level1):
            continue
        for id_dir in os.listdir(level1):
            level2 = os.path.join(level1, id_dir)
            if not os.path.isdir(level2):
                continue
            nid = _normalize_id(id_dir)
            for source_name, dest_suffix in SOURCE_TO_DEST.items():
                source_path = _find_file_in_dir(level2, source_name)
                if source_path is None:
                    continue
                dest_name = f"{nid}{dest_suffix}"
                dest_dir = os.path.dirname(source_path)
                dest_path = os.path.join(dest_dir, dest_name)
                if os.path.isfile(dest_path):
                    continue
                xyz_rows = _parse_rows_to_xyz(source_path)
                if xyz_rows is None:
                    continue
                _write_xyz_rgb(dest_path, xyz_rows, *RED_RGB)
