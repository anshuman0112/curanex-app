"""Scan upload (ALL_MODELS_PLAN Route A, 2D first): a DICOM file, or a .zip of one series, becomes one plain image.

Only the pixels are kept. The DICOM header, which holds the name, ID, birth date and hospital, is never saved: the
image is re-rendered from the pixel array and the original bytes are dropped, so every identifying tag is removed.
Files that say identifying text is burned into the pixels are refused, because that text would survive.

For a series, the middle slice is used, which is what a single-slice 2D model expects. The display window stored in
the file (what a radiologist sees) is used when present; otherwise the 1st-99th percentile range.
"""
import io
import zipfile

import numpy as np
from PIL import Image

MAX_FILES = 1000
MAX_UNZIPPED = 300 * 1024 * 1024
MODALITY = {"CT": "CT scan", "MR": "MRI scan", "CR": "X-ray", "DX": "X-ray", "DR": "X-ray", "MG": "Mammogram", "US": "Ultrasound"}


class ScanRejected(ValueError):
    """Shown to the user as is."""


def is_scan(data: bytes) -> bool:
    return data[128:132] == b"DICM" or data[:4] == b"PK\x03\x04"


def _read(data: bytes):
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        return pydicom.dcmread(io.BytesIO(data), force=data[128:132] != b"DICM")
    except (InvalidDicomError, EOFError, ValueError, OSError):
        return None


def _series(data: bytes) -> list:
    if data[128:132] == b"DICM":
        ds = _read(data)
        if ds is None:
            raise ScanRejected("That DICOM file couldn't be read.")
        return [ds]
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ScanRejected("That .zip file couldn't be opened.") from e
    files = [i for i in z.infolist() if not i.is_dir()]
    if len(files) > MAX_FILES or sum(i.file_size for i in files) > MAX_UNZIPPED:
        raise ScanRejected("That .zip is too large. Upload one series (up to 1000 slices).")
    out = []
    for info in files:
        if info.filename.startswith("__MACOSX") or info.filename.rsplit("/", 1)[-1].startswith("."):
            continue
        ds = _read(z.read(info))
        if ds is not None and "PixelData" in ds:
            out.append(ds)
    if not out:
        raise ScanRejected("No DICOM images were found in that .zip.")
    return out


def _middle(series: list):
    uid = series[0].get("SeriesInstanceUID")
    same = [d for d in series if d.get("SeriesInstanceUID") == uid] or series

    def where(d):
        pos = d.get("ImagePositionPatient")
        return float(pos[2]) if pos is not None and len(pos) == 3 else float(d.get("InstanceNumber") or 0)

    same.sort(key=where)
    return same[len(same) // 2], len(same)


def _render(ds) -> np.ndarray:
    try:
        px = ds.pixel_array.astype(np.float32)
    except Exception as e:  # noqa: BLE001 - compressed formats need extra decoders
        raise ScanRejected("This DICOM uses a compression the server can't open. Export it as uncompressed DICOM, PNG or JPEG.") from e
    if px.ndim == 3 and px.shape[-1] not in (3, 4):  # multi-frame: middle frame
        px = px[px.shape[0] // 2]
    if px.ndim == 3:
        return np.clip(px[..., :3], 0, 255).astype(np.uint8)
    px = px * float(ds.get("RescaleSlope", 1) or 1) + float(ds.get("RescaleIntercept", 0) or 0)
    center, width = ds.get("WindowCenter"), ds.get("WindowWidth")
    if center is not None and width is not None:
        c = float(center[0] if hasattr(center, "__len__") and not isinstance(center, str) else center)
        w = float(width[0] if hasattr(width, "__len__") and not isinstance(width, str) else width)
        lo, hi = c - w / 2, c + w / 2
    else:
        lo, hi = np.percentile(px, [1, 99])
    out = np.clip((px - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        out = 1 - out
    return (out * 255).astype(np.uint8)


def scan_to_image(data: bytes) -> tuple[bytes, dict]:
    """Returns a PNG of the representative slice plus non-identifying facts about it (modality, slice count)."""
    ds, count = _middle(_series(data))
    if str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
        raise ScanRejected("This image has patient details printed on it, so it can't be uploaded. Use an anonymised export.")
    img = Image.fromarray(_render(ds))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    modality = str(ds.get("Modality", "")).upper()
    return buf.getvalue(), {"source": "dicom", "modality": MODALITY.get(modality, modality or "Scan"), "slices": count}
