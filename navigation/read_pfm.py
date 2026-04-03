from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_pfm(file_path: str | Path) -> tuple[np.ndarray, float]:
    path = Path(file_path)
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dimensions = handle.readline().decode("utf-8")
        match = re.match(r"^(\d+)\s+(\d+)\s*$", dimensions)
        if not match:
            raise ValueError(f"Malformed PFM dimensions in {path!s}: {dimensions!r}")
        width, height = map(int, match.groups())

        scale = float(handle.readline().decode("utf-8").rstrip())
        if scale < 0:
            endian = "<"
            scale = -scale
        else:
            endian = ">"

        data = np.fromfile(handle, endian + "f")
        expected = width * height * (3 if color else 1)
        if data.size != expected:
            raise ValueError(f"PFM size mismatch in {path}: expected {expected}, got {data.size}")

        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.astype(np.float32), scale
