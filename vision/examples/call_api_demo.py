"""
Demo script showing how the backend or other microservices interact with the running
Vision FastAPI service (GET /vision/health, POST /vision/process-frame, POST /vision/estimate).
Posts each of the 12 sample frames in vision/demo_assets/ and displays the returned OccupancyUpdate.
"""

import json
from pathlib import Path
import sys
import httpx

_assets_dir = Path(__file__).resolve().parent.parent / "demo_assets"
BASE_URL = "http://127.0.0.1:8000"


def main():
    print("=" * 70)
    print("CALLING VISION & FUSION MICROSERVICE API")
    print(f"Target Service URL: {BASE_URL}")
    print("=" * 70)

    # 1. Health check
    try:
        r_health = httpx.get(f"{BASE_URL}/vision/health", timeout=5.0)
        print(f"\n[1] GET /vision/health -> HTTP {r_health.status_code}")
        print(json.dumps(r_health.json(), indent=2))
        if r_health.status_code != 200 or not r_health.json().get("model_ready"):
            print("Warning: Vision model is not ready according to health endpoint.")
    except Exception as exc:
        print(f"[Error] Failed to connect to vision service at {BASE_URL}: {exc}")
        print("Please start the service with: uvicorn vision.src.main_vision:app --port 8000")
        sys.exit(1)

    # 2. Iterate through demo assets and call /vision/process-frame
    jpg_files = sorted(list(_assets_dir.glob("*.jpg")))
    print(f"\n[2] POST /vision/process-frame for {len(jpg_files)} demo frames:")
    print("-" * 70)

    for idx, jpg_path in enumerate(jpg_files, 1):
        json_path = jpg_path.with_suffix(".json")
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            zone_cfg = json.load(f)

        with open(jpg_path, "rb") as img_f:
            files = {"image": (jpg_path.name, img_f, "image/jpeg")}
            data = {"zone_config": json.dumps(zone_cfg)}

            r_frame = httpx.post(f"{BASE_URL}/vision/process-frame", files=files, data=data, timeout=15.0)

            if r_frame.status_code == 200:
                up = r_frame.json()
                print(f"[{idx:02d}/12] {jpg_path.name:<25} | Fac: {up['facility_id']:<10} | Cap: {up['capacity']:<3} | Free: [{up['free_low']}, {up['free_high']}] | Conf: {up['confidence']:.2f}")
            else:
                print(f"[{idx:02d}/12] {jpg_path.name:<25} | Error HTTP {r_frame.status_code}: {r_frame.text}")

    # 3. Call /vision/estimate
    print("\n[3] POST /vision/estimate (Sensor Fusion):")
    estimate_payload = {
        "zone": {
            "facility_id": "FAC-PUCPR",
            "zone_id": "ZONE-PUCPR",
            "capacity": 100,
            "zone_type": "managed"
        },
        "observations": {
            "camera": {
                "timestamp": "2026-11-15T07:00:00Z",
                "count": 45
            },
            "manual": {
                "timestamp": "2026-11-15T07:10:00Z",
                "count": 48
            }
        },
        "now": "2026-11-15T07:12:00Z"
    }
    r_est = httpx.post(f"{BASE_URL}/vision/estimate", json=estimate_payload, timeout=5.0)
    print(f"HTTP {r_est.status_code}:")
    print(json.dumps(r_est.json(), indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()

