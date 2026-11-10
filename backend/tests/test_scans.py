"""Scan upload: DICOM and zipped series become one image, identifying tags never reach storage, bad files are refused."""
import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from api.uploads_store import load_image
from main import app

client = TestClient(app)
NAME = "Sharma^Ravi"


def dicom(z: float = 0.0, value: int = 1240, burned: bool = False, series: str = "1.2.3") -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("x.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID, ds.PatientBirthDate, ds.InstitutionName = NAME, "MRN-55501", "19800101", "City Hospital"
    ds.Modality, ds.SeriesInstanceUID, ds.ImagePositionPatient = "CT", series, [0, 0, z]
    ds.Rows = ds.Columns = 64
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleSlope, ds.RescaleIntercept, ds.WindowCenter, ds.WindowWidth = 1, -1024, 40, 400
    px = np.zeros((64, 64), np.int16)
    px[16:48, 16:48] = value
    ds.PixelData = px.tobytes()
    if burned:
        ds.BurnedInAnnotation = "YES"
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_a_dicom_becomes_an_image_without_its_header():
    r = client.post("/upload", files={"file": ("scan.dcm", dicom(), "application/dicom")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "dicom" and body["modality"] == "CT scan" and body["slices"] == 1
    saved = load_image(body["image_id"])
    assert saved and b"Sharma" not in saved and b"MRN-55501" not in saved and b"City Hospital" not in saved
    img = np.asarray(Image.open(io.BytesIO(saved)).convert("L"))
    assert img[32, 32] > 200 and img[2, 2] < 30  # the window was applied


def test_a_zipped_series_uses_the_middle_slice():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, z_pos in enumerate([30.0, 10.0, 20.0]):
            z.writestr(f"s{i}.dcm", dicom(z_pos, value=1024 + int(z_pos) * 10))
    r = client.post("/upload", files={"file": ("series.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 200 and r.json()["slices"] == 3


def test_burned_in_details_are_refused():
    r = client.post("/upload", files={"file": ("scan.dcm", dicom(burned=True), "application/dicom")})
    assert r.status_code == 415 and "printed" in r.json()["detail"]


@pytest.mark.parametrize("payload", [b"PK\x03\x04garbage", b"\0" * 128 + b"DICM" + b"\0" * 20])
def test_broken_scans_are_refused(payload):
    r = client.post("/upload", files={"file": ("x", payload, "application/octet-stream")})
    assert r.status_code == 415
