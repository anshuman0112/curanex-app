"""Session image store. Uploads are validated, re-encoded (which drops EXIF, including GPS) and kept by random id."""
import io
import re
import uuid

from PIL import Image, UnidentifiedImageError

from api.core.config import get_settings

ALLOWED = {"JPEG", "PNG", "WEBP"}
_ID = re.compile(r"^[0-9a-f]{32}$")


class UploadRejected(Exception):
    pass


def save_image(data: bytes) -> str:
    s = get_settings()
    if len(data) > s.upload_max_bytes:
        raise UploadRejected(f"File is larger than {s.upload_max_bytes // (1024 * 1024)} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise UploadRejected("Not a supported image (JPEG, PNG or WebP)") from e
    if img.format not in ALLOWED:
        raise UploadRejected("Not a supported image (JPEG, PNG or WebP)")
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    image_id = uuid.uuid4().hex
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    img.save(s.uploads_dir / f"{image_id}.jpg", format="JPEG", quality=95)  # fresh file: no EXIF
    return image_id


def load_image(image_id: str) -> bytes | None:
    if not _ID.match(image_id or ""):
        return None
    path = get_settings().uploads_dir / f"{image_id}.jpg"
    return path.read_bytes() if path.exists() else None
