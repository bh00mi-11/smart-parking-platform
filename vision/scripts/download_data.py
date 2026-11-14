"""
Dataset acquisition and verification script for the Vision Module.
Downloads:
1. CNRPark-Patches (official 150x150 patch dataset from GitHub archive release).
2. Deep-parking splits archive (containing official splits for PKLot and CNRPark-EXT).
3. MetaPKLot-Dataset spot annotations (complete ground-truth polygons, rotatedRects,
   occupancies, climates, and timestamps for all 12,417 images across PUCPR, UFPR04, UFPR05).
4. Representative full-frame PKLot images across all three lots (PUCPR, UFPR04, UFPR05)
   and weather conditions (Sunny, Cloudy, Rainy) directly with corresponding per-slot XMLs.
"""

import os
import sys
import io
import lzma
import tarfile
import zipfile
import json
import argparse
from collections import defaultdict
from typing import Optional
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
from tqdm import tqdm
import socket

# Fallback DNS resolution for UFPR server if local Windows DNS fails
_original_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, port, *args, **kwargs):
    if host in ['www.inf.ufpr.br', 'whiskey.c3sl.ufpr.br']:
        return _original_getaddrinfo('200.17.202.113', port, *args, **kwargs)
    return _original_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = _patched_getaddrinfo

PKLOT_OFFICIAL_URL = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"

CNRPARK_PATCHES_URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip"
SPLITS_URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/splits.zip"
METAPKLOT_BASE_URL = "https://raw.githubusercontent.com/DSBD-Research/MetaPKLot-Dataset/main/annotations/original/spots/PKLot/"

SAMPLE_IMAGES_MANIFEST = [
    # (lot, weather, date, filename, hf_data_folder)
    ("PUCPR", "Cloudy", "2012-10-16", "2012-10-16_06_21_43.jpg", "data_2"),
    ("PUCPR", "Sunny", "2012-09-11", "2012-09-11_15_27_08.jpg", "data_0"),
    ("PUCPR", "Rainy", "2012-10-25", "2012-10-25_19_33_54.jpg", "data_3"),
    ("PUCPR", "Sunny", "2012-09-11", "2012-09-11_15_16_58.jpg", "data_0"),
    ("UFPR04", "Sunny", "2012-12-26", "2012-12-26_08_25_03.jpg", "data_5"),
    ("UFPR04", "Cloudy", "2012-12-22", "2012-12-22_15_55_12.jpg", "data_7"),
    ("UFPR04", "Rainy", "2012-12-28", "2012-12-28_09_05_04.jpg", "data_8"),
    ("UFPR04", "Rainy", "2012-12-07", "2012-12-07_16_42_25.jpg", "data_8"),
    ("UFPR05", "Sunny", "2013-04-16", "2013-04-16_07_45_02.jpg", "data_10"),
    ("UFPR05", "Cloudy", "2013-04-12", "2013-04-12_15_35_10.jpg", "data_11"),
    ("UFPR05", "Rainy", "2013-04-14", "2013-04-14_15_30_10.jpg", "data_9"),
    ("UFPR05", "Sunny", "2013-02-22", "2013-02-22_06_05_00.jpg", "data_9"),
]

def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[{desc}] File already exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))
    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def setup_cnrpark(data_dir: Path):
    cnr_dir = data_dir / "CNRPark"
    if (cnr_dir / "A").exists():
        print("[CNRPark] Directory already prepared.")
        return
    cnr_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "CNRPark-Patches-150x150.zip"
    download_file(CNRPARK_PATCHES_URL, zip_path, "CNRPark Patches")
    print(f"[CNRPark] Extracting to {cnr_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cnr_dir)
    print("[CNRPark] Extraction completed.")

def setup_splits(data_dir: Path):
    splits_dir = data_dir / "splits"
    if splits_dir.exists() and (splits_dir / "PKLot").exists():
        print("[Splits] Splits directory already prepared.")
        return
    splits_zip = data_dir / "splits.zip"
    if not splits_zip.exists():
        # Check parent folder
        parent_zip = data_dir.parent / "splits.zip"
        if parent_zip.exists():
            parent_zip.rename(splits_zip)
        else:
            download_file(SPLITS_URL, splits_zip, "Official Splits")
    print(f"[Splits] Extracting splits to {data_dir}...")
    with zipfile.ZipFile(splits_zip, "r") as z:
        z.extractall(data_dir)
    print("[Splits] Extraction completed.")

def setup_metapklot_annotations(data_dir: Path):
    anno_dir = data_dir / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    for lot_key in ["pucpr", "ufpr04", "ufpr05"]:
        json_file = anno_dir / f"{lot_key}_spots.json"
        if json_file.exists():
            continue
        tar_url = f"{METAPKLOT_BASE_URL}{lot_key}_spots.tar.xz"
        print(f"[MetaPKLot] Fetching annotations for {lot_key}...")
        resp = requests.get(tar_url, timeout=40)
        resp.raise_for_status()
        decomp = lzma.decompress(resp.content)
        with tarfile.open(fileobj=io.BytesIO(decomp)) as tar:
            with open(json_file, "wb") as f_out:
                f_out.write(tar.extractfile(f"{lot_key}_spots.json").read())
        print(f"[MetaPKLot] Saved {json_file}")

def synthesize_xml_from_annotation(image_record: dict, annotations: list, xml_path: Path):
    """
    Synthesizes standard PKLot XML structure from spot annotations.
    Matches the official schema:
    <parking id="...">
      <space id="..." occupied="...">
        <rotatedRect>
          <center x="..." y="..." />
          <size w="..." h="..." />
          <angle d="..." />
        </rotatedRect>
        <contour>
          <point x="..." y="..." />
          ...
        </contour>
      </space>
    </parking>
    """
    parking_elem = ET.Element("parking", id=image_record["subset"])
    for idx, anno in enumerate(annotations, 1):
        space_id = str(anno.get("car_id", idx))
        occupied_str = str(anno["category_id"])
        space_elem = ET.SubElement(parking_elem, "space", id=space_id, occupied=occupied_str)

        rot = anno.get("best_rotated_rect")
        if rot and len(rot) >= 3:
            (cx, cy), (w, h), angle = rot[0], rot[1], rot[2]
            rot_elem = ET.SubElement(space_elem, "rotatedRect")
            ET.SubElement(rot_elem, "center", x=str(round(cx)), y=str(round(cy)))
            ET.SubElement(rot_elem, "size", w=str(round(w)), h=str(round(h)))
            ET.SubElement(rot_elem, "angle", d=str(round(angle)))

        seg = anno.get("segmentation", [])
        if seg and len(seg[0]) >= 6:
            pts = seg[0]
            contour_elem = ET.SubElement(space_elem, "contour")
            for i in range(0, len(pts), 2):
                ET.SubElement(contour_elem, "point", x=str(round(pts[i])), y=str(round(pts[i+1])))

    tree = ET.ElementTree(parking_elem)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def setup_pklot_full_frames(data_dir: Path):
    pklot_dir = data_dir / "PKLot"
    pklot_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations for PUCPR, UFPR04, UFPR05
    annos_by_lot = {}
    for lot_key in ["pucpr", "ufpr04", "ufpr05"]:
        json_file = data_dir / "annotations" / f"{lot_key}_spots.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Map filename -> list of annotations
            img_map = {img["file_name"].split("/")[-1]: (img, []) for img in data["images"]}
            for anno in data["annotations"]:
                img_id = anno["image_id"]
            # Fast index by img_id
            id_to_fname = {img["id"]: img["file_name"].split("/")[-1] for img in data["images"]}
            for anno in data["annotations"]:
                fname = id_to_fname.get(anno["image_id"])
                if fname and fname in img_map:
                    img_map[fname][1].append(anno)
            annos_by_lot[lot_key] = img_map

    for lot, weather, date, fname, hf_folder in SAMPLE_IMAGES_MANIFEST:
        dest_jpg = pklot_dir / lot / weather / date / fname
        dest_xml = pklot_dir / lot / weather / date / fname.replace(".jpg", ".xml")
        if dest_jpg.exists() and dest_xml.exists():
            continue

        dest_jpg.parent.mkdir(parents=True, exist_ok=True)
        img_url = f"https://huggingface.co/datasets/Voxel51/PKLot/resolve/main/data/{hf_folder}/{fname}"
        print(f"[PKLot] Downloading {lot}/{weather}/{fname}...")
        download_file(img_url, dest_jpg, f"Image {fname}")

        # Generate XML
        lot_key = lot.lower()
        if lot_key in annos_by_lot and fname in annos_by_lot[lot_key]:
            img_record, annos = annos_by_lot[lot_key][fname]
            synthesize_xml_from_annotation(img_record, annos, dest_xml)
            print(f"[PKLot] Synthesized XML for {fname} with {len(annos)} slot annotations.")
        else:
            print(f"[PKLot] Warning: No spot annotation found for {fname}")

def extract_sampled_pklot_archive(
    archive_path: Path,
    pklot_dir: Path,
    frames_per_lot: int = 150
):
    """
    Extracts a stratified, evenly sampled subset of PKLot full-frame images
    and XMLs across lots (PUCPR, UFPR04, UFPR05), weathers, and days.
    Supports .tar.gz / .tar and .zip archives.
    Requires each selected .jpg to have a corresponding .xml.
    """
    pklot_dir.mkdir(parents=True, exist_ok=True)
    is_tar = str(archive_path).endswith((".tar.gz", ".tgz", ".tar"))
    is_zip = str(archive_path).endswith(".zip")

    if not is_tar and not is_zip:
        raise ValueError(f"Unsupported archive format: {archive_path.name}. Must be .tar.gz, .tar, or .zip")

    # 1. Inspect archive members and list first 20 names
    print(f"[PKLot] Reading members from {archive_path.name}...")
    if is_tar:
        archive_obj = tarfile.open(archive_path, "r:*")
        all_member_names = [m.name for m in archive_obj.getmembers() if not m.isdir()]
    else:
        archive_obj = zipfile.ZipFile(archive_path, "r")
        all_member_names = [m for m in archive_obj.namelist() if not m.endswith("/")]

    print("\n--- Detected Archive Member Sample (First 20 Names) ---")
    for name in all_member_names[:20]:
        print(f"  {name}")
    print("------------------------------------------------------\n")

    # Map members into set of all member names for fast XML lookup
    member_name_set = set(all_member_names)

    # 2. Group full-frame jpg files having corresponding xml files
    # Structure: groups[lot][(weather, day)] -> list of member paths
    groups = defaultdict(lambda: defaultdict(list))
    total_valid_pairs = 0
    missing_xml_count = 0
    detected_lots = set()

    for name in all_member_names:
        if name.endswith(".jpg"):
            xml_name = name[:-4] + ".xml"
            if xml_name not in member_name_set:
                missing_xml_count += 1
                continue

            # Identify lot, weather, day
            parts = Path(name).parts
            lot = None
            for candidate in ["PUCPR", "UFPR04", "UFPR05"]:
                if candidate in parts:
                    lot = candidate
                    break
            if lot is not None:
                idx = parts.index(lot)
                if len(parts) >= idx + 4:
                    weather = parts[idx + 1]
                    day = parts[idx + 2]
                    groups[lot][(weather, day)].append(name)
                    total_valid_pairs += 1
                    detected_lots.add(lot)

    print(f"[PKLot] Folder layout parsed. Found {total_valid_pairs} valid jpg+xml pairs across lots: {sorted(list(detected_lots))}")
    if missing_xml_count > 0:
        print(f"[PKLot] Warning: Skipped {missing_xml_count} .jpg files that had no corresponding .xml")

    # Check that all required lots have frames
    required_lots = ["PUCPR", "UFPR04", "UFPR05"]
    for lot in required_lots:
        lot_frame_count = sum(len(lst) for lst in groups[lot].values())
        if lot_frame_count == 0:
            archive_obj.close()
            raise RuntimeError(
                f"[PKLot Sampler Error] Lot '{lot}' has 0 valid jpg+xml frames in archive! "
                f"Available lots: {sorted(list(detected_lots))}"
            )

    # 3. Stratified Stride Sampling per (lot, weather, day)
    selected_jpg_members = []
    print("\n=== Sampling PKLot Frames per Lot, Weather, and Day ===")
    for lot in required_lots:
        weather_days = groups[lot]
        lot_chosen = []

        # Sort weather and day combinations deterministically
        sorted_keys = sorted(weather_days.keys())
        all_lot_frames = []
        for k in sorted_keys:
            weather_days[k].sort()
            all_lot_frames.extend(weather_days[k])

        total_frames = len(all_lot_frames)
        target = min(frames_per_lot, total_frames)

        # Calculate stride per (weather, day) bucket or across flattened sorted list
        if total_frames > target:
            # Stride evenly across the flattened ordered frames
            stride = total_frames / float(target)
            for i in range(target):
                idx = int(i * stride)
                lot_chosen.append(all_lot_frames[idx])
        else:
            lot_chosen = list(all_lot_frames)

        # Print breakdown per weather and day for this lot
        chosen_by_wd = defaultdict(int)
        for chosen_name in lot_chosen:
            parts = Path(chosen_name).parts
            idx = parts.index(lot)
            w = parts[idx + 1]
            d = parts[idx + 2]
            chosen_by_wd[(w, d)] += 1

        print(f"\nLot '{lot}': Selected {len(lot_chosen)} / {total_frames} frames (target {frames_per_lot}):")
        for (w, d) in sorted(chosen_by_wd.keys()):
            avail = len(weather_days[(w, d)])
            print(f"  - Weather: {w:<8} | Day: {d:<12} | Sampled: {chosen_by_wd[(w, d)]:<3} / {avail:<3} frames")

        selected_jpg_members.extend(lot_chosen)

    # Verify that every selected jpg has its xml in the archive
    selected_member_set = set()
    for jpg_name in selected_jpg_members:
        xml_name = jpg_name[:-4] + ".xml"
        if xml_name not in member_name_set:
            archive_obj.close()
            raise RuntimeError(f"[PKLot Sampler Error] Image {jpg_name} is missing corresponding XML {xml_name} in archive!")
        selected_member_set.add(jpg_name)
        selected_member_set.add(xml_name)

    print(f"\n[PKLot] Extracting {len(selected_member_set)} files ({len(selected_jpg_members)} pairs) to {pklot_dir}...")

    # 4. Extract ONLY the sampled pairs directly into pklot_dir
    if is_tar:
        for member in tqdm(archive_obj.getmembers(), desc="Extracting sampled PKLot"):
            if member.name in selected_member_set:
                parts = Path(member.name).parts
                for candidate in ["PUCPR", "UFPR04", "UFPR05"]:
                    if candidate in parts:
                        idx = parts.index(candidate)
                        rel_path = Path(*parts[idx:])
                        out_target = pklot_dir / rel_path
                        out_target.parent.mkdir(parents=True, exist_ok=True)
                        f_in = archive_obj.extractfile(member)
                        if f_in is not None:
                            with open(out_target, "wb") as f_out:
                                f_out.write(f_in.read())
                        break
        archive_obj.close()
    else:
        for m_name in tqdm(selected_member_set, desc="Extracting sampled PKLot"):
            parts = Path(m_name).parts
            for candidate in ["PUCPR", "UFPR04", "UFPR05"]:
                if candidate in parts:
                    idx = parts.index(candidate)
                    rel_path = Path(*parts[idx:])
                    out_target = pklot_dir / rel_path
                    out_target.parent.mkdir(parents=True, exist_ok=True)
                    with archive_obj.open(m_name, "r") as f_in:
                        with open(out_target, "wb") as f_out:
                            f_out.write(f_in.read())
                    break
        archive_obj.close()

    print(f"[PKLot] Successfully extracted {len(selected_jpg_members)} pairs to {pklot_dir}")


def download_and_extract_pklot_archive(
    data_dir: Path,
    archive_path: Optional[Path] = None,
    frames_per_lot: int = 150
):
    """
    Downloads or uses an existing archive, then runs extract_sampled_pklot_archive.
    """
    pklot_dir = data_dir / "PKLot"
    pklot_dir.mkdir(parents=True, exist_ok=True)

    if archive_path is not None:
        target_archive = Path(archive_path)
        if not target_archive.exists():
            raise FileNotFoundError(f"Specified archive not found: {target_archive}")
        print(f"[PKLot] Using existing local archive: {target_archive} (Download skipped)")
    else:
        target_archive = data_dir / "PKLot.tar.gz"
        print("=" * 65)
        print("PKLot Official Archive Downloader & Sampler")
        print(f"Target frames per lot: {frames_per_lot}")
        print("=" * 65)
        download_file(PKLOT_OFFICIAL_URL, target_archive, "PKLot Official Archive (4.56 GB)")

    extract_sampled_pklot_archive(target_archive, pklot_dir, frames_per_lot=frames_per_lot)


def main():
    parser = argparse.ArgumentParser(description="Download and setup parking datasets")
    parser.add_argument("--official-archive", action="store_true", help="Download and sample official PKLot.tar.gz archive (for Colab)")
    parser.add_argument("--archive-path", type=str, default=None, help="Path to existing local PKLot archive (.tar.gz, .tar, or .zip) to sample without downloading")
    parser.add_argument("--frames-per-lot", type=int, default=150, help="Frames to sample per lot from official PKLot archive")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.official_archive or args.archive_path:
        print("=== Official PKLot Archive Sampler Mode ===")
        archive_p = Path(args.archive_path) if args.archive_path else None
        download_and_extract_pklot_archive(data_dir, archive_path=archive_p, frames_per_lot=args.frames_per_lot)
    else:
        print("=== Lightweight Sample Mode (Local Development) ===")
        setup_metapklot_annotations(data_dir)
        setup_pklot_full_frames(data_dir)
        setup_cnrpark(data_dir)
        setup_splits(data_dir)

    print("=== Dataset setup successfully completed ===")

if __name__ == "__main__":
    main()
