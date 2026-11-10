"""
Unit tests for PKLot archive sampler in vision/scripts/download_data.py.
Builds a synthetic .tar.gz archive in the exact PKLot layout, runs the sampler,
and verifies extraction parity, counts, and error conditions.
"""

import io
import tarfile
from pathlib import Path
import pytest

from vision.scripts.download_data import extract_sampled_pklot_archive


def make_synthetic_pklot_tar(archive_path: Path, lots=None, missing_xml_for=None):
    if lots is None:
        lots = ["PUCPR", "UFPR04", "UFPR05"]

    dummy_jpg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb"
    dummy_xml_bytes = b"""<?xml version="1.0" encoding="UTF-8"?>
<parking id="test">
  <space id="1" occupied="0">
    <rotatedRect>
      <center x="50" y="50" />
      <size w="30" h="60" />
      <angle d="15" />
    </rotatedRect>
  </space>
</parking>"""

    with tarfile.open(archive_path, "w:gz") as tar:
        for lot in lots:
            for weather in ["Sunny", "Cloudy", "Rainy"]:
                for day_idx in range(1, 3):
                    date_str = f"2026-10-0{day_idx}"
                    for img_idx in range(1, 4):
                        base_name = f"{lot}_{date_str}_{img_idx:03d}"
                        # Typical PKLot archive layout: PKLot/PKLot/{lot}/{weather}/{date_str}/{base_name}.jpg
                        rel_jpg = f"PKLot/PKLot/{lot}/{weather}/{date_str}/{base_name}.jpg"
                        rel_xml = f"PKLot/PKLot/{lot}/{weather}/{date_str}/{base_name}.xml"

                        # Add JPG
                        tarinfo_jpg = tarfile.TarInfo(name=rel_jpg)
                        tarinfo_jpg.size = len(dummy_jpg_bytes)
                        tar.addfile(tarinfo_jpg, io.BytesIO(dummy_jpg_bytes))

                        # Conditionally add XML
                        if missing_xml_for and missing_xml_for in rel_jpg:
                            continue

                        tarinfo_xml = tarfile.TarInfo(name=rel_xml)
                        tarinfo_xml.size = len(dummy_xml_bytes)
                        tar.addfile(tarinfo_xml, io.BytesIO(dummy_xml_bytes))


def test_sampler_extracts_correct_pairs_and_counts(tmp_path):
    archive_path = tmp_path / "PKLot_mock.tar.gz"
    out_pklot_dir = tmp_path / "extracted_PKLot"

    # 3 lots * 3 weathers * 2 days * 3 imgs = 18 frames per lot, 54 frames total
    make_synthetic_pklot_tar(archive_path)

    # Sample 5 frames per lot
    target_frames_per_lot = 5
    extract_sampled_pklot_archive(
        archive_path=archive_path,
        pklot_dir=out_pklot_dir,
        frames_per_lot=target_frames_per_lot
    )

    # Assert all 3 lots exist in extracted directory
    for lot in ["PUCPR", "UFPR04", "UFPR05"]:
        lot_dir = out_pklot_dir / lot
        assert lot_dir.exists(), f"Lot directory {lot} was not created!"

        jpg_files = list(lot_dir.rglob("*.jpg"))
        xml_files = list(lot_dir.rglob("*.xml"))

        # Exact count matches target
        assert len(jpg_files) == target_frames_per_lot, (
            f"Expected {target_frames_per_lot} jpgs for lot {lot}, got {len(jpg_files)}"
        )
        assert len(xml_files) == target_frames_per_lot, (
            f"Expected {target_frames_per_lot} xmls for lot {lot}, got {len(xml_files)}"
        )

        # Assert every single JPG has its matching XML
        for jpg in jpg_files:
            xml_match = jpg.with_suffix(".xml")
            assert xml_match.exists(), f"JPG {jpg} is missing corresponding XML {xml_match}!"


def test_sampler_raises_error_if_lot_has_zero_frames(tmp_path):
    archive_path = tmp_path / "PKLot_missing_lot.tar.gz"
    out_pklot_dir = tmp_path / "extracted_PKLot"

    # Only include PUCPR and UFPR04 (UFPR05 is missing)
    make_synthetic_pklot_tar(archive_path, lots=["PUCPR", "UFPR04"])

    with pytest.raises(RuntimeError, match="has 0 valid jpg\\+xml frames"):
        extract_sampled_pklot_archive(
            archive_path=archive_path,
            pklot_dir=out_pklot_dir,
            frames_per_lot=5
        )
