import math
import os
import re

import pandas as pd

# Point file names (none are required; missing or unparseable files are reasons a parameter may be impossible)
SEAT_CUSHION_POINTS_FILE = "SeatCushion_Points.txt"
SEAT_POINTS_FILE = "Seat_Points.txt"
FRONT_WHEEL_POINTS_FILE = "FrontWheel_Points.txt"
REAR_WHEEL_POINTS_FILE = "RearWheel_Points.txt"
# Any point with a coordinate <= this value, or exactly 0.0, is treated as disformed
DISFORMED_COORD_THRESHOLD = -999.0


def _normalize_id(dir_name: str) -> str:
    """Normalize directory name to match Excel ID format: remove '_', then strip after the first letter+number part."""
    s = dir_name.replace("_", "")
    match = re.match(r"^([A-Za-z]+\d+).*", s)
    return match.group(1) if match else s


def _find_file_in_dir(root_path: str, filename: str) -> str | None:
    """Return full path to filename under root_path, or None if not found."""
    for dirpath, _dirs, files in os.walk(root_path):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def _parse_points_file(path: str) -> tuple[dict[str, tuple[float, float, float]] | None, str | None]:
    """
    Parse a points file. Each row: index, label, x, y, z (whitespace/tab delimited).
    Returns (dict mapping label -> (x, y, z), None) on success, or (None, path) if malformed.
    """
    out: dict[str, tuple[float, float, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                label = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                return (None, path)
            out[label] = (x, y, z)
    return (out, None)


def _parse_points_file_ordered(
    path: str,
) -> tuple[list[tuple[str, tuple[float, float, float]]] | None, str | None]:
    """
    Parse a points file, preserving row order. Returns (list of (label, (x,y,z)), None) on success, or (None, path) if malformed.
    """
    out: list[tuple[str, tuple[float, float, float]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                label = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                return (None, path)
            out.append((label, (x, y, z)))
    return (out, None)


def _has_disformed_coord(point: tuple[float, float, float]) -> bool:
    """True if any coordinate is <= DISFORMED_COORD_THRESHOLD or exactly 0.0."""
    return any(c <= DISFORMED_COORD_THRESHOLD or c == 0.0 for c in point)


def _dist3d(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(
        (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
    )


def _point_to_plane_distance(
    point: tuple[float, float, float],
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    """
    Signed distance from point to the plane through p0, p1, p2.
    We return the absolute value so panHeight is positive.
    """
    # Vectors in the plane
    v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    # Normal n = v1 x v2
    n = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    n_len = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if n_len == 0:
        return 0.0
    # Vector from p0 to point
    diff = (point[0] - p0[0], point[1] - p0[1], point[2] - p0[2])
    dist = abs(n[0] * diff[0] + n[1] * diff[1] + n[2] * diff[2]) / n_len
    return dist


def _try_dist(
    d: dict[str, tuple[float, float, float]], label1: str, label2: str, scale: float = 1.0
) -> float | None:
    """Return scale * distance between label1 and label2 if both exist and are valid; else None."""
    if label1 not in d or label2 not in d:
        return None
    p, q = d[label1], d[label2]
    if _has_disformed_coord(p) or _has_disformed_coord(q):
        return None
    return _dist3d(p, q) * scale


def _try_seat_width(
    cushion: dict[str, tuple[float, float, float]] | None,
    seat: dict[str, tuple[float, float, float]] | None,
) -> float | None:
    """Try seatWidth methods in order; return first that succeeds. Uses only non-None data."""
    if cushion is None and seat is None:
        return None
    methods: list[tuple[dict[str, tuple[float, float, float]] | None, str, str, float]] = [
        (cushion, "SC1", "SC2", 1.0),
        (cushion, "SC4", "SC5", 1.0),
        (cushion, "SC2", "SC3", 2.0),
        (cushion, "SC1", "SC3", 2.0),
        (seat, "SC6", "SC8", 1.0),
        (seat, "SC9", "SC10", 1.0),
        (seat, "SC6", "SC7", 2.0),
        (seat, "SC7", "SC8", 2.0),
    ]
    for d, l1, l2, scale in methods:
        if d is not None:
            v = _try_dist(d, l1, l2, scale)
            if v is not None:
                return v
    return None


def _try_seat_depth(
    cushion: dict[str, tuple[float, float, float]] | None,
    seat: dict[str, tuple[float, float, float]] | None,
) -> float | None:
    """Try seatDepth methods in order; return first that succeeds. Uses only non-None data."""
    if cushion is None and seat is None:
        return None
    methods: list[tuple[dict[str, tuple[float, float, float]] | None, str, str]] = [
        (cushion, "SC2", "SC4"),
        (cushion, "SC1", "SC5"),
        (seat, "SC8", "SC10"),
        (seat, "SC6", "SC9"),
    ]
    for d, l1, l2 in methods:
        if d is not None:
            v = _try_dist(d, l1, l2)
            if v is not None:
                return v
    return None


def _try_pan_height(
    seat_ordered: list[tuple[str, tuple[float, float, float]]] | None,
    front_wheel: dict[str, tuple[float, float, float]] | None,
    rear_wheel: dict[str, tuple[float, float, float]] | None = None,
) -> float | None:
    """Plane from first 3 valid in Seat_Points; try distance to WFL2, WFL3, WFR2 (FrontWheel), then WRL2, WRL4, WRR2, WRR4 (RearWheel if present)."""
    if seat_ordered is None:
        return None
    plane_pts: list[tuple[float, float, float]] = []
    for _label, pt in seat_ordered:
        if not _has_disformed_coord(pt):
            plane_pts.append(pt)
            if len(plane_pts) >= 3:
                break
    if len(plane_pts) < 3:
        return None
    if front_wheel is None and rear_wheel is None:
        return None
    front_labels = ("WFL2", "WFL3", "WFR2", "WFR3")
    rear_labels = ("WRL2", "WRL4", "WRR2", "WRR4")
    if front_wheel is not None:
        for label in front_labels:
            if label not in front_wheel or _has_disformed_coord(front_wheel[label]):
                continue
            d = _point_to_plane_distance(
                front_wheel[label], plane_pts[0], plane_pts[1], plane_pts[2]
            )
            return d
    if rear_wheel is not None:
        for label in rear_labels:
            if label not in rear_wheel or _has_disformed_coord(rear_wheel[label]):
                continue
            d = _point_to_plane_distance(
                rear_wheel[label], plane_pts[0], plane_pts[1], plane_pts[2]
            )
            return d
    return None


def _compute_measurements_multi(
    cushion: dict[str, tuple[float, float, float]] | None,
    seat: dict[str, tuple[float, float, float]] | None,
    seat_ordered: list[tuple[str, tuple[float, float, float]]] | None,
    front_wheel: dict[str, tuple[float, float, float]] | None,
    rear_wheel: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    """
    Try each calculation method in order for seatWidth, seatDepth, panHeight.
    Missing or invalid data (e.g. no files) can make a parameter impossible to calculate.
    Returns (seat_width, seat_depth, pan_height, invalid_reason).
    """
    sw = _try_seat_width(cushion, seat)
    sd = _try_seat_depth(cushion, seat)
    ph = _try_pan_height(seat_ordered, front_wheel, rear_wheel)
    if sw is None:
        return (None, None, None, "seatWidth")
    if sd is None:
        return (None, None, None, "seatDepth")
    if ph is None:
        return (None, None, None, "panHeight")
    return (sw, sd, ph, None)


def _compute_measurements(
    cushion_path: str | None,
    seat_path: str | None,
    front_wheel_path: str | None,
    rear_wheel_path: str | None = None,
) -> tuple[tuple[float, float, float] | None, list[str], str | None]:
    """
    Load whatever point files exist and compute seatWidth, seatDepth, panHeight.
    Missing or unparseable files are reasons a parameter may be impossible to calculate.
    Returns (result, disformed_paths, invalid_reason). On success: ((sw,sd,ph), [], None).
    """
    disformed: list[str] = []
    cushion: dict[str, tuple[float, float, float]] | None = None
    seat: dict[str, tuple[float, float, float]] | None = None
    seat_ordered: list[tuple[str, tuple[float, float, float]]] | None = None
    front_wheel: dict[str, tuple[float, float, float]] | None = None
    rear_wheel: dict[str, tuple[float, float, float]] | None = None
    if cushion_path is not None:
        c, err = _parse_points_file(cushion_path)
        if err is not None:
            disformed.append(err)
        else:
            cushion = c
    if seat_path is not None:
        s, err = _parse_points_file(seat_path)
        if err is not None:
            disformed.append(err)
        else:
            seat = s
        so, err = _parse_points_file_ordered(seat_path)
        if err is not None and err not in disformed:
            disformed.append(err)
        elif so is not None:
            seat_ordered = so
    if front_wheel_path is not None:
        fw, err = _parse_points_file(front_wheel_path)
        if err is not None:
            disformed.append(err)
        else:
            front_wheel = fw
    if rear_wheel_path is not None:
        rw, err = _parse_points_file(rear_wheel_path)
        if err is not None:
            disformed.append(err)
        else:
            rear_wheel = rw
    sw, sd, ph, invalid_reason = _compute_measurements_multi(
        cushion, seat, seat_ordered, front_wheel, rear_wheel
    )
    if invalid_reason is not None:
        return (None, disformed, invalid_reason)
    return ((sw, sd, ph), disformed, None)


def _count_valid_points(ordered_points: list[tuple[str, tuple[float, float, float]]]) -> int:
    """Return how many points have no coordinate <= DISFORMED_COORD_THRESHOLD."""
    return sum(1 for _label, pt in ordered_points if not _has_disformed_coord(pt))


def count_dirs_both_seat_files_malformed(dir_path: str = "digitizations") -> int:
    """
    Count ID-level directories where both Seat_Points.txt and SeatCushion_Points.txt
    are malformed (fewer than 3 valid points in each). Prints the count and returns it.
    """
    if not os.path.isdir(dir_path):
        print("Directories with both Seat and SeatCushion malformed (< 3 valid points each): 0")
        return 0
    count = 0
    for name in os.listdir(dir_path):
        level1 = os.path.join(dir_path, name)
        if not os.path.isdir(level1):
            continue
        for id_dir in os.listdir(level1):
            level2 = os.path.join(level1, id_dir)
            if not os.path.isdir(level2):
                continue
            seat_path = _find_file_in_dir(level2, SEAT_POINTS_FILE)
            cushion_path = _find_file_in_dir(level2, SEAT_CUSHION_POINTS_FILE)
            if seat_path is None or cushion_path is None:
                continue
            seat_ordered, err = _parse_points_file_ordered(seat_path)
            if err is not None:
                seat_valid = 0
            else:
                seat_valid = _count_valid_points(seat_ordered)
            cushion_ordered, err = _parse_points_file_ordered(cushion_path)
            if err is not None:
                cushion_valid = 0
            else:
                cushion_valid = _count_valid_points(cushion_ordered)
            if seat_valid < 3 and cushion_valid < 3:
                count += 1
    print(f"Directories with both Seat_Points and SeatCushion_Points malformed (< 3 valid points each): {count}")
    return count


class DigitizationsReader:
    """Builds a DataFrame of IDs and seat/wheel measurements from digitizations/(name)/ID directory structure."""

    def __init__(self):
        self.df: pd.DataFrame | None = None

    def load(
        self,
        dir_path: str = "digitizations",
        allowed_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        Walk directory 2 levels down (digitizations/(name)/ID). Only directories whose normalized ID
        is in allowed_ids (if provided) are considered. For each directory that contains all three
        required .txt files, try calculation methods in order; if all methods fail for a parameter,
        invalidate the directory. Returns DataFrame with ID, seatWidth, seatDepth, panHeight.
        """
        rows: list[dict] = []
        seen: set[str] = set()
        disformed_paths: set[str] = set()
        total_dirs = 0
        invalidated_reason: dict[str, str] = {}  # nid -> reason; cleared when that nid succeeds
        if not os.path.isdir(dir_path):
            self.df = pd.DataFrame(columns=["ID", "seatWidth", "seatDepth", "panHeight"])
            return self.df
        for name in os.listdir(dir_path):
            level1 = os.path.join(dir_path, name)
            if not os.path.isdir(level1):
                continue
            for id_dir in os.listdir(level1):
                level2 = os.path.join(level1, id_dir)
                if not os.path.isdir(level2):
                    continue
                nid = _normalize_id(id_dir)
                if allowed_ids is not None and nid not in allowed_ids:
                    continue
                if nid in seen:
                    continue
                cushion_path = _find_file_in_dir(level2, SEAT_CUSHION_POINTS_FILE)
                seat_path = _find_file_in_dir(level2, SEAT_POINTS_FILE)
                front_wheel_path = _find_file_in_dir(level2, FRONT_WHEEL_POINTS_FILE)
                rear_wheel_path = _find_file_in_dir(level2, REAR_WHEEL_POINTS_FILE)
                total_dirs += 1
                result, disformed, invalid_reason = _compute_measurements(
                    cushion_path, seat_path, front_wheel_path, rear_wheel_path
                )
                for p in disformed:
                    print(p)
                    disformed_paths.add(p)
                if invalid_reason is not None:
                    invalidated_reason[nid] = invalid_reason
                    continue
                if result is None:
                    continue
                invalidated_reason.pop(nid, None)
                seen.add(nid)
                seat_width, seat_depth, pan_height = result
                rows.append({
                    "ID": nid,
                    "seatWidth": seat_width,
                    "seatDepth": seat_depth,
                    "panHeight": pan_height,
                })
        if disformed_paths:
            print(f"Total disformed txt files: {len(disformed_paths)}")
        for nid, reason in sorted(invalidated_reason.items()):
            print(f"{nid} was invalidated because {reason} was impossible to calculate")
        if total_dirs > 0:
            print(f"Directories invalidated (parameter impossible to calculate): {len(invalidated_reason)} / {total_dirs}")
        if not rows:
            self.df = pd.DataFrame(columns=["ID", "seatWidth", "seatDepth", "panHeight"])
        else:
            self.df = pd.DataFrame(rows).astype({"seatWidth": float, "seatDepth": float, "panHeight": float})
        return self.df
