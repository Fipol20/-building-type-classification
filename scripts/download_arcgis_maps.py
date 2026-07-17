"""Скачивание карт с ArcGIS Online REST Services.

Источник: https://server.arcgisonline.com/ArcGIS/rest/services/

Примеры:
  python scripts/download_arcgis_maps.py --list
  python scripts/download_arcgis_maps.py \\
      --service World_Imagery/MapServer \\
      --center 37.62,55.76 \\
      --zoom 16 \\
      --size 512,512 \\
      --out data/maps/moscow_512.jpg

  python scripts/download_arcgis_maps.py \\
      --mode export \\
      --service World_Street_Map/MapServer \\
      --bbox 2.29,48.85,2.36,48.88 \\
      --size 512,512 \\
      --out data/maps/paris_streets.png

Использование карт Esri регулируется их Terms of Use:
https://www.esri.com/en-us/legal/terms/full-master-agreement
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "maps"
BASE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/"
TILE_SIZE = 256
DEFAULT_SIZE = (512, 512)
DEFAULT_SERVICE = "World_Imagery/MapServer"

KNOWN_SERVICES = {
    "World_Imagery/MapServer": "Спутниковые снимки (World Imagery)",
    "World_Street_Map/MapServer": "Улицы (World Street Map)",
    "World_Topo_Map/MapServer": "Топографическая карта",
    "World_Physical_Map/MapServer": "Физическая карта",
    "World_Shaded_Relief/MapServer": "Рельеф",
    "World_Terrain_Base/MapServer": "Рельеф + база",
    "USA_Topo_Maps/MapServer": "Топокарты США",
    "NatGeo_World_Map/MapServer": "National Geographic",
}


def lon_lat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = lon * 20_037_508.342_789_244 / 180.0
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) / math.radians(1)
    y = y * 20_037_508.342_789_244 / 180.0
    return x, y


def mercator_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    lon = x * 180.0 / 20_037_508.342_789_244
    lat = math.degrees(2 * math.atan(math.exp(y * math.pi / 20_037_508.342_789_244)) - math.pi / 2)
    return lon, lat


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox: lon_min,lat_min,lon_max,lat_max")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise argparse.ArgumentTypeError("bbox: min < max по lon и lat")
    return lon_min, lat_min, lon_max, lat_max


def parse_center(raw: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("center: lon,lat")
    return parts[0], parts[1]


def parse_size(raw: str) -> tuple[int, int]:
    parts = [int(x.strip()) for x in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size: width,height")
    width, height = parts
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size: width и height > 0")
    return width, height


def service_url(service: str, suffix: str = "") -> str:
    service = service.strip("/")
    return urljoin(BASE_URL, f"{service}/{suffix}".strip("/"))


def fetch_json(url: str, session: requests.Session, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params={"f": "pjson"}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Не удалось получить {url}: {last_error}")


def list_services(session: requests.Session, folder: str = "") -> list[tuple[str, str]]:
    url = urljoin(BASE_URL, folder)
    data = fetch_json(url, session)
    items: list[tuple[str, str]] = []

    for subfolder in data.get("folders", []):
        items.append((subfolder, "folder"))
        items.extend(list_services(session, f"{subfolder}/"))

    for svc in data.get("services", []):
        name = svc["name"]
        kind = svc.get("type", "Unknown")
        items.append((name, kind))

    return items


def print_service_catalog(session: requests.Session) -> None:
    print(f"ArcGIS REST Services: {BASE_URL}\n")
    print("Популярные сервисы:")
    for path, desc in KNOWN_SERVICES.items():
        print(f"  {path:<35} {desc}")

    print("\nПолный каталог:")
    for name, kind in sorted(list_services(session)):
        prefix = "[folder]" if kind == "folder" else kind
        print(f"  {name:<50} {prefix}")


def get_lod_resolution(meta: dict[str, Any], zoom: int) -> float:
    lods = meta["tileInfo"]["lods"]
    by_level = {lod["level"]: lod["resolution"] for lod in lods}
    if zoom in by_level:
        return by_level[zoom]
    if 0 <= zoom < len(lods):
        return lods[zoom]["resolution"]
    raise ValueError(f"zoom {zoom} вне диапазона 0..{max(by_level)}")


def bbox_from_center(
    center: tuple[float, float],
    size: tuple[int, int],
    zoom: int,
    meta: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Bbox WGS84 ровно под size x size пикселей на заданном zoom."""
    lon, lat = center
    x, y = lon_lat_to_mercator(lon, lat)
    resolution = get_lod_resolution(meta, zoom)
    half_w = size[0] * resolution / 2
    half_h = size[1] * resolution / 2
    lon_min, lat_min = mercator_to_lon_lat(x - half_w, y - half_h)
    lon_max, lat_max = mercator_to_lon_lat(x + half_w, y + half_h)
    return lon_min, lat_min, lon_max, lat_max


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def save_image(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(out_path, quality=95)
    else:
        image.save(out_path)


def tile_range_for_bbox(
    bbox_merc: tuple[float, float, float, float],
    zoom: int,
    meta: dict[str, Any],
) -> tuple[int, int, int, int, float, float, float]:
    xmin, ymin, xmax, ymax = bbox_merc
    origin = meta["tileInfo"]["origin"]
    origin_x, origin_y = origin["x"], origin["y"]
    resolution = get_lod_resolution(meta, zoom)
    tile_span = TILE_SIZE * resolution

    col_min = math.floor((xmin - origin_x) / tile_span)
    col_max = math.floor((xmax - origin_x) / tile_span)
    row_min = math.floor((origin_y - ymax) / tile_span)
    row_max = math.floor((origin_y - ymin) / tile_span)

    mosaic_origin_x = origin_x + col_min * tile_span
    mosaic_origin_y = origin_y - row_min * tile_span
    return col_min, col_max, row_min, row_max, resolution, mosaic_origin_x, mosaic_origin_y


def download_tile(
    service: str,
    zoom: int,
    row: int,
    col: int,
    retries: int = 3,
) -> Image.Image:
    url = service_url(service, f"tile/{zoom}/{row}/{col}")
    headers = {"User-Agent": "building-type-classification/arcgis-downloader"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"tile {zoom}/{row}/{col}: {last_error}")


def download_tiles_mosaic(
    service: str,
    bbox_wgs84: tuple[float, float, float, float],
    zoom: int,
    out_path: Path,
    size: tuple[int, int],
    tiles_dir: Path | None = None,
    workers: int = 8,
) -> dict[str, int | float]:
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    xmin, ymin = lon_lat_to_mercator(lon_min, lat_min)
    xmax, ymax = lon_lat_to_mercator(lon_max, lat_max)
    bbox_merc = (xmin, ymin, xmax, ymax)

    session = requests.Session()
    session.headers.update({"User-Agent": "building-type-classification/arcgis-downloader"})
    meta = fetch_json(service_url(service), session)

    if not meta.get("singleFusedMapCache"):
        raise RuntimeError(
            f"{service} не поддерживает tile cache. Используйте --mode export."
        )

    col_min, col_max, row_min, row_max, resolution, mosaic_x, mosaic_y = tile_range_for_bbox(
        bbox_merc, zoom, meta
    )
    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    total = n_cols * n_rows
    print(
        f"Сервис: {service}\n"
        f"Zoom: {zoom}, resolution: {resolution:.2f} m/px\n"
        f"Тайлы: cols {col_min}..{col_max}, rows {row_min}..{row_max} ({total} шт.)"
    )

    if tiles_dir is not None:
        tiles_dir.mkdir(parents=True, exist_ok=True)

    mosaic = Image.new("RGB", (n_cols * TILE_SIZE, n_rows * TILE_SIZE))
    downloaded = 0

    tasks: list[tuple[int, int]] = [
        (row, col)
        for row in range(row_min, row_max + 1)
        for col in range(col_min, col_max + 1)
    ]

    def _fetch(row: int, col: int) -> tuple[int, int, Image.Image]:
        tile = download_tile(service, zoom, row, col)
        if tiles_dir is not None:
            tile.save(tiles_dir / f"{zoom}_{row}_{col}.jpg", quality=95)
        return row, col, tile

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_fetch, row, col) for row, col in tasks]
        for future in as_completed(futures):
            row, col, tile = future.result()
            px = (col - col_min) * TILE_SIZE
            py = (row - row_min) * TILE_SIZE
            mosaic.paste(tile, (px, py))
            downloaded += 1
            if downloaded % 20 == 0 or downloaded == total:
                print(f"  загружено {downloaded}/{total}")

    left = int(round((xmin - mosaic_x) / resolution))
    top = int(round((mosaic_y - ymax) / resolution))
    right = int(round((xmax - mosaic_x) / resolution))
    bottom = int(round((mosaic_y - ymin) / resolution))
    cropped = mosaic.crop((left, top, right, bottom))
    output = resize_image(cropped, size)
    save_image(output, out_path)
    print(f"Сохранено: {out_path} ({output.width}x{output.height})")

    return {
        "tiles": total,
        "width": output.width,
        "height": output.height,
        "resolution_m": resolution,
    }


def download_export_image(
    service: str,
    bbox_wgs84: tuple[float, float, float, float],
    size: tuple[int, int],
    out_path: Path,
) -> dict[str, int]:
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    xmin, ymin = lon_lat_to_mercator(lon_min, lat_min)
    xmax, ymax = lon_lat_to_mercator(lon_max, lat_max)

    session = requests.Session()
    session.headers.update({"User-Agent": "building-type-classification/arcgis-downloader"})
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{size[0]},{size[1]}",
        "format": "png",
        "f": "image",
    }
    url = service_url(service, "export")
    resp = session.get(url, params=params, timeout=120)
    resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"Сохранено: {out_path} ({size[0]}x{size[1]})")
    return {"width": size[0], "height": size[1]}


def save_metadata(out_path: Path, payload: dict[str, Any]) -> None:
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачивание карт с ArcGIS Online REST Services.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "bbox задаётся в WGS84: lon_min,lat_min,lon_max,lat_max.\n"
            f"Каталог сервисов: {BASE_URL}"
        ),
    )
    parser.add_argument("--list", action="store_true", help="Показать доступные сервисы")
    parser.add_argument(
        "--service",
        default=DEFAULT_SERVICE,
        help=f"Путь сервиса, напр. {DEFAULT_SERVICE}",
    )
    parser.add_argument(
        "--mode",
        choices=("tiles", "export"),
        default="tiles",
        help="tiles — тайлы + склейка; export — один снимок через /export",
    )
    parser.add_argument("--bbox", type=parse_bbox, help="lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--center", type=parse_center, help="lon,lat — центр патча (вместо bbox)")
    parser.add_argument("--zoom", type=int, default=16, help="Уровень тайлов (mode=tiles)")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=f"{DEFAULT_SIZE[0]},{DEFAULT_SIZE[1]}",
        help=f"Размер выхода width,height (по умолчанию {DEFAULT_SIZE[0]},{DEFAULT_SIZE[1]})",
    )
    parser.add_argument("--out", type=Path, help="Путь к выходному файлу")
    parser.add_argument("--tiles-dir", type=Path, help="Сохранять отдельные тайлы")
    parser.add_argument("--workers", type=int, default=8, help="Параллельных загрузок")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "building-type-classification/arcgis-downloader"})

    if args.list:
        print_service_catalog(session)
        return

    if args.bbox is None and args.center is None:
        parser.error("Укажите --bbox или --center (или --list)")

    if args.bbox is not None and args.center is not None:
        parser.error("Используйте либо --bbox, либо --center")

    bbox = args.bbox
    if args.center is not None:
        meta = fetch_json(service_url(args.service), session)
        bbox = bbox_from_center(args.center, args.size, args.zoom, meta)
        print(
            f"Центр {args.center[0]:.6f},{args.center[1]:.6f} -> "
            f"bbox {bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
        )

    out_path = args.out
    if out_path is None:
        stem = args.service.replace("/", "_").lower()
        suffix = ".png" if args.mode == "export" else ".jpg"
        out_path = DEFAULT_OUT_DIR / f"{stem}_{args.size[0]}x{args.size[1]}_z{args.zoom}{suffix}"

    payload: dict[str, Any] = {
        "source": BASE_URL,
        "service": args.service,
        "mode": args.mode,
        "size": list(args.size),
        "bbox_wgs84": {
            "lon_min": bbox[0],
            "lat_min": bbox[1],
            "lon_max": bbox[2],
            "lat_max": bbox[3],
        },
    }
    if args.center is not None:
        payload["center"] = {"lon": args.center[0], "lat": args.center[1]}

    if args.mode == "tiles":
        stats = download_tiles_mosaic(
            service=args.service,
            bbox_wgs84=bbox,
            zoom=args.zoom,
            out_path=out_path,
            size=args.size,
            tiles_dir=args.tiles_dir,
            workers=args.workers,
        )
        payload.update({"zoom": args.zoom, **stats})
    else:
        stats = download_export_image(
            service=args.service,
            bbox_wgs84=bbox,
            size=args.size,
            out_path=out_path,
        )
        payload.update(**stats)

    save_metadata(out_path, payload)
    print(f"Метаданные: {out_path.with_suffix(out_path.suffix + '.json')}")


if __name__ == "__main__":
    main()
