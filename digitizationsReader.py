import math
import os
import re
import numpy as np
import pandas as pd

# Point file names
SEAT_CUSHION_POINTS_FILE = "SeatCushion_Points.txt"
SEAT_POINTS_FILE = "Seat_Points.txt"
FRONT_WHEEL_POINTS_FILE = "FrontWheel_Points.txt"
REAR_WHEEL_POINTS_FILE = "RearWheel_Points.txt"
# Any point with a coordinate <= this value, or exactly 0.0, is treated as disformed
DISFORMED_COORD_THRESHOLD = -999.0
# Minimum valid points per cloud for unified logic
MIN_POINTS = 3
# Smallest extent (mm) for non-degenerate cushion box; wheel plane must span 2D
DEGENERACY_EPS = 1.0
# Width candidates with length in [seatDepth * (0.5 - HALF_WIDTH_TOL), seatDepth * (0.5 + HALF_WIDTH_TOL)] are excluded (center-to-corner segment)
HALF_WIDTH_TOL = 0.15


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


def _pca_3d(
    points: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None:
    """
    PCA on 3D point cloud. Returns (centroid, axes_3x3, extents_3) with extents descending.
    axes[i] is the i-th principal direction; extents[i] is extent along that axis.
    Returns None if fewer than 3 points or degenerate (covariance rank < 3).
    """
    if len(points) < 3:
        return None
    X = np.array(points, dtype=float)
    centroid = np.mean(X, axis=0)
    C = X - centroid
    cov = (C.T @ C) / (len(C) - 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    # ascending order from eigh; reverse so largest first
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Extents: project points onto each axis and take max - min
    proj = C @ eigenvectors
    extents = np.array(
        [np.ptp(proj[:, i]) for i in range(3)],
        dtype=float,
    )
    return (tuple(centroid.tolist()), eigenvectors, extents)


def _fit_plane_normal(
    points: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fit a plane to points via PCA. Normal = direction of smallest variance (smallest eigenvector).
    Returns (normal_unit, centroid) or None if collinear (middle eigenvalue too small).
    """
    if len(points) < 3:
        return None
    X = np.array(points, dtype=float)
    centroid = np.mean(X, axis=0)
    C = X - centroid
    cov = (C.T @ C) / max(len(C) - 1, 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    # Smallest eigenvalue -> normal direction
    normal = eigenvectors[:, 0].copy()
    n_len = np.linalg.norm(normal)
    if n_len < 1e-10:
        return None
    normal /= n_len
    # Non-collinear: second eigenvalue must be above threshold (points span a plane)
    if eigenvalues[1] < DEGENERACY_EPS**2:
        return None
    return (normal, centroid)


def _pairwise_candidates(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, np.ndarray, tuple[float, float, float], tuple[float, float, float]]]:
    """Return list of (extent, direction_unit, p1, p2) for all pairs i < j."""
    n = len(points)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = np.array(points[i])
            p2 = np.array(points[j])
            v = p2 - p1
            d = np.linalg.norm(v)
            if d < 1e-10:
                continue
            out.append((float(d), v / d, points[i], points[j]))
    return out


def _write_seat_pcd_debug(
    id_str: str,
    out_dir: str,
    combined: list[tuple[float, float, float]],
    centroid: np.ndarray,
    axes: np.ndarray,
    axis_name: str,
    axis_index: int,
) -> None:
    """Write two points (min/max along axis) to {id_str}_seatPCD_{axis_name}.txt as x y z r g b (red)."""
    axis_vec = axes[:, axis_index]
    proj = [np.dot(np.array(pt) - centroid, axis_vec) for pt in combined]
    min_idx = int(np.argmin(proj))
    max_idx = int(np.argmax(proj))
    p_min = combined[min_idx]
    p_max = combined[max_idx]
    path = os.path.join(out_dir, f"{id_str}_seatPCD_{axis_name}.txt")
    with open(path, "w") as f:
        f.write(f"{p_min[0]} {p_min[1]} {p_min[2]} 255 0 0\n")
        f.write(f"{p_max[0]} {p_max[1]} {p_max[2]} 255 0 0\n")


def _write_two_points_debug(
    id_str: str, out_dir: str, axis_name: str,
    p1: tuple[float, float, float], p2: tuple[float, float, float],
) -> None:
    """Write two points to {id_str}_seatPCD_{axis_name}.txt as x y z 255 0 0."""
    path = os.path.join(out_dir, f"{id_str}_seatPCD_{axis_name}.txt")
    with open(path, "w") as f:
        f.write(f"{p1[0]} {p1[1]} {p1[2]} 255 0 0\n")
        f.write(f"{p2[0]} {p2[1]} {p2[2]} 255 0 0\n")


def _compute_unified(
    cushion_top: list[tuple[float, float, float]],
    cushion_bottom: list[tuple[float, float, float]],
    front_wheel: list[tuple[float, float, float]],
    rear_wheel: list[tuple[float, float, float]],
    debug_id: str | None = None,
    debug_out_dir: str | None = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    """
    Unified seatWidth, seatDepth, panHeight from four point clouds.
    - Depth axis from wheel plane (front or rear, whichever has more points).
    - Thickness direction = direction of shortest vector over all top-bottom cushion point pairs.
    - Width axis = cross(thickness_dir, depth_axis) normalized.
    - Width/depth extents = pairwise candidates from one cushion cloud; choose best-aligned (then longest) to depth_axis and width_axis.
    - Pan height = distance from seat plane (bottom centroid, normal thickness_dir) to furthest wheel point below.
    Returns (seat_width, seat_depth, pan_height, invalid_reason). invalid_reason is None on success.
    If debug_id and debug_out_dir are set, writes {ID}_seatPCD_{width,depth,thickness}.txt with the two points chosen per axis (x y z 255 0 0).
    """
    # Prereq: at least 3 valid points in each cushion file
    if len(cushion_top) < MIN_POINTS:
        return (None, None, None, "seatWidth")
    if len(cushion_bottom) < MIN_POINTS:
        return (None, None, None, "seatDepth")
    # At least one wheel cloud has >= 3 points; use the one with more valid points when both valid
    use_front = len(front_wheel) >= MIN_POINTS
    use_rear = len(rear_wheel) >= MIN_POINTS
    if not use_front and not use_rear:
        return (None, None, None, "panHeight")

    # Depth axis from wheel plane: fit plane to a single cloud (prefer the one with more points)
    if use_front and use_rear:
        wheel_for_plane = front_wheel if len(front_wheel) >= len(rear_wheel) else rear_wheel
    elif use_front:
        wheel_for_plane = front_wheel
    else:
        wheel_for_plane = rear_wheel
    plane_result = _fit_plane_normal(wheel_for_plane)
    if plane_result is None:
        return (None, None, None, "panHeight")
    depth_axis, _ = plane_result
    depth_axis = np.asarray(depth_axis, dtype=float)

    # Thickness direction: shortest vector among all top-bottom pairs (robust when top/bottom points not corresponding)
    best_len = np.inf
    thickness_dir = None
    thickness_p1: tuple[float, float, float] | None = None
    thickness_p2: tuple[float, float, float] | None = None
    for t in cushion_top:
        for b in cushion_bottom:
            v = np.array(t) - np.array(b)
            d = np.linalg.norm(v)
            if d < 1e-10:
                continue
            if d < best_len:
                best_len = d
                thickness_dir = v / d
                thickness_p1 = b
                thickness_p2 = t
    if thickness_dir is None:
        return (None, None, None, "panHeight")
    thickness_dir = thickness_dir.astype(float)

    # Width axis: in-plane, perpendicular to depth (cross of thickness and depth)
    width_axis = np.cross(thickness_dir, depth_axis)
    w_len = np.linalg.norm(width_axis)
    if w_len < 1e-10:
        return (None, None, None, "seatWidth")
    width_axis = width_axis / w_len

    # Width/depth from pairwise candidates in a single cloud (avoids diagonal being chosen as an extent)
    in_plane_cloud = cushion_top if len(cushion_top) >= len(cushion_bottom) else cushion_bottom
    candidates = _pairwise_candidates(in_plane_cloud)
    if not candidates:
        return (None, None, None, "seatWidth")

    # Depth extent: candidate that aligns best with depth_axis; among ties, longest
    best_depth_align = -1.0
    seat_depth = -1.0
    depth_p1: tuple[float, float, float] | None = None
    depth_p2: tuple[float, float, float] | None = None
    for ext, direction, p1, p2 in candidates:
        align = abs(float(np.dot(direction, depth_axis)))
        if align > best_depth_align or (align == best_depth_align and ext > seat_depth):
            best_depth_align = align
            seat_depth = ext
            depth_p1, depth_p2 = p1, p2

    # Width extent: candidate that aligns best with width_axis; among ties, longest.
    # Exclude candidates whose length is close to half of seatDepth (center-to-corner segment).
    half_lo = seat_depth * (0.5 - HALF_WIDTH_TOL)
    half_hi = seat_depth * (0.5 + HALF_WIDTH_TOL)
    best_width_align = -1.0
    seat_width = -1.0
    width_p1: tuple[float, float, float] | None = None
    width_p2: tuple[float, float, float] | None = None
    for ext, direction, p1, p2 in candidates:
        if half_lo <= ext <= half_hi:
            continue
        align = abs(float(np.dot(direction, width_axis)))
        if align > best_width_align or (align == best_width_align and ext > seat_width):
            best_width_align = align
            seat_width = ext
            width_p1, width_p2 = p1, p2

    if seat_width < 0 or seat_depth < 0:
        return (None, None, None, "seatWidth")

    bottom_centroid = np.mean(np.array(cushion_bottom), axis=0)
    # Seat plane: through bottom centroid, normal = thickness_dir (pointing "up")
    seat_plane_origin = bottom_centroid
    all_wheel = front_wheel + rear_wheel
    if not all_wheel:
        return (None, None, None, "panHeight")
    # Signed distance from each wheel point to seat plane (positive = same side as thickness_dir)
    pan_heights: list[float] = []
    for pt in all_wheel:
        diff = np.array(pt) - seat_plane_origin
        signed_dist = float(np.dot(diff, thickness_dir))
        pan_heights.append(signed_dist)
    # Point furthest below the seat plane (most negative) -> pan height = abs(that distance)
    min_signed = min(pan_heights)
    pan_height = abs(min_signed)

    # Debug: write the two points chosen for each axis
    if debug_id is not None and debug_out_dir is not None:
        if width_p1 is not None and width_p2 is not None:
            _write_two_points_debug(debug_id, debug_out_dir, "width", width_p1, width_p2)
        if depth_p1 is not None and depth_p2 is not None:
            _write_two_points_debug(debug_id, debug_out_dir, "depth", depth_p1, depth_p2)
        if thickness_p1 is not None and thickness_p2 is not None:
            _write_two_points_debug(debug_id, debug_out_dir, "thickness", thickness_p1, thickness_p2)

    return (seat_width, seat_depth, pan_height, None)


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


def _valid_points_from_parsed(d: dict[str, tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Return list of valid (non-disformed) points from a parsed label->point dict."""
    return [pt for pt in d.values() if not _has_disformed_coord(pt)]


def _compute_measurements(
    cushion_path: str | None,
    seat_path: str | None,
    front_wheel_path: str | None,
    rear_wheel_path: str | None = None,
    debug_id: str | None = None,
    debug_out_dir: str | None = None,
) -> tuple[tuple[float, float, float] | None, list[str], str | None]:
    """
    Load point files, extract valid points (label-free), and compute seatWidth, seatDepth, panHeight
    via unified PCA + wheel-plane logic. Prereqs: SeatCushion and Seat_Points each >= 3 valid points,
    at least one wheel file with >= 3 valid points; non-degenerate cushion box and wheel plane.
    Returns (result, disformed_paths, invalid_reason). On success: ((sw,sd,ph), [], None).
    If debug_id and debug_out_dir are set, writes {ID}_seatPCD_{width,depth,thickness}.txt for debugging.
    """
    disformed: list[str] = []
    cushion_top: list[tuple[float, float, float]] = []
    cushion_bottom: list[tuple[float, float, float]] = []
    front_wheel: list[tuple[float, float, float]] = []
    rear_wheel: list[tuple[float, float, float]] = []

    if cushion_path is not None:
        parsed, err = _parse_points_file(cushion_path)
        if err is not None:
            disformed.append(err)
        elif parsed is not None:
            cushion_top = _valid_points_from_parsed(parsed)

    if seat_path is not None:
        parsed, err = _parse_points_file(seat_path)
        if err is not None:
            if err not in disformed:
                disformed.append(err)
        elif parsed is not None:
            cushion_bottom = _valid_points_from_parsed(parsed)

    if front_wheel_path is not None:
        parsed, err = _parse_points_file(front_wheel_path)
        if err is not None:
            if err not in disformed:
                disformed.append(err)
        elif parsed is not None:
            front_wheel = _valid_points_from_parsed(parsed)

    if rear_wheel_path is not None:
        parsed, err = _parse_points_file(rear_wheel_path)
        if err is not None:
            if err not in disformed:
                disformed.append(err)
        elif parsed is not None:
            rear_wheel = _valid_points_from_parsed(parsed)

    sw, sd, ph, invalid_reason = _compute_unified(
        cushion_top, cushion_bottom, front_wheel, rear_wheel,
        debug_id=debug_id,
        debug_out_dir=debug_out_dir,
    )
    if invalid_reason is not None:
        return (None, disformed, invalid_reason)
    assert sw is not None and sd is not None and ph is not None
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
        is in allowed_ids (if provided) are considered. Uses unified PCA + wheel-plane logic on all
        four point clouds (SeatCushion, Seat_Points, FrontWheel, RearWheel). Prereqs: >= 3 valid
        points in each cushion file, >= 3 in at least one wheel file; non-degenerate cushion box
        and wheel plane. Returns DataFrame with ID, seatWidth, seatDepth, panHeight.
        Always writes debug {ID}_seatPCD_{width,depth,thickness}.txt into each ID's directory (two points per axis, x y z 255 0 0).
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
                    cushion_path, seat_path, front_wheel_path, rear_wheel_path,
                    debug_id=nid,
                    debug_out_dir=level2,
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
