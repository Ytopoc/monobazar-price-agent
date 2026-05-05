# Test script: send a photo to the API.
import base64
import json
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000/api/v1/price"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_with_photo.py <image_path> [category_id] [description]")
        print("Example: python scripts/test_with_photo.py photo.jpg 4 'iPhone 13 128GB'")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    category_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    description = sys.argv[3] if len(sys.argv) > 3 else ""

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"Image: {image_path} ({len(b64) // 1024} KB base64)")
    print(f"Category: {category_id}")
    print(f"Description: {description or '(empty — will use photo only)'}")
    print()

    payload = {
        "description": description,
        "photos": [b64],
    }
    if category_id is not None:
        payload["category_id"] = category_id

    print("Sending request...")
    resp = requests.post(API_URL, json=payload, timeout=60)

    print(f"Status: {resp.status_code}")
    print()

    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
