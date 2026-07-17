"""Слияние карты застройки по районам (zone, 4 класса) и классов зданий (building, 3 класса).

Правило слияния:
  - building-класс commercial/industrial -> итоговый класс = как есть, зона не влияет
  - building-класс residential -> смотрим zone-класс по центроиду bbox здания:
      * зона dense_residential/sparse_residential -> итоговый класс = класс зоны
      * иначе (зона commercial/industrial, либо zone_map отсутствует)
        -> итоговый класс = residential (нерешённый/смешанный участок)
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from building_masks import Building
from dataset import CLASSES as ZONE_MAP_CLASSES
from utils import ZONE_COLORS
from zone_map import zone_class_at, zone_probs_at

UNRESOLVED_CLASS = "residential"
RESIDENTIAL_ZONE_CLASSES = {"dense_residential", "sparse_residential"}

# Палитра для итоговой (merged) визуализации: 4 zone-класса + residential (нерешённый)
MERGED_COLORS: dict[str, tuple[int, int, int]] = {
    **ZONE_COLORS,
    UNRESOLVED_CLASS: (46, 184, 92),
}


def merge_zone_and_buildings(
    buildings: list[Building],
    zone_map: np.ndarray | None,
    zone_classes: list[str] = ZONE_MAP_CLASSES,
    zone_prob_map: np.ndarray | None = None,
    zone_residential_min_prob: float = 0.0,
) -> list[Building]:
    """Заполняет `final_class`/`merge_note` у каждого здания по правилу слияния."""
    dense_idx = zone_classes.index("dense_residential") if "dense_residential" in zone_classes else None
    sparse_idx = zone_classes.index("sparse_residential") if "sparse_residential" in zone_classes else None

    for b in buildings:
        if b.pred_class is None:
            b.final_class = None
            b.merge_note = "отфильтровано: низкая уверенность class"
            continue

        cls = b.final_source_class

        if cls in ("commercial", "industrial"):
            b.final_class = cls
            b.merge_note = "building-класс commercial/industrial, зона не учитывается"
            continue

        cx, cy = b.centroid

        if zone_prob_map is not None and dense_idx is not None and sparse_idx is not None:
            probs = zone_probs_at(zone_prob_map, cx, cy)
            if probs is not None:
                p_dense = float(probs[dense_idx])
                p_sparse = float(probs[sparse_idx])
                if p_dense >= p_sparse and p_dense >= zone_residential_min_prob:
                    b.final_class = "dense_residential"
                    b.merge_note = f"residential; P(dense)={p_dense:.2f} >= {zone_residential_min_prob:.2f}"
                    continue
                if p_sparse > p_dense and p_sparse >= zone_residential_min_prob:
                    b.final_class = "sparse_residential"
                    b.merge_note = f"residential; P(sparse)={p_sparse:.2f} >= {zone_residential_min_prob:.2f}"
                    continue
                zone_cls = zone_classes[int(np.argmax(probs))]
                b.final_class = UNRESOLVED_CLASS
                b.merge_note = (
                    f"residential; max(P_dense,P_sparse) < {zone_residential_min_prob:.2f}, "
                    f"argmax zone={zone_cls}"
                )
                continue

        if zone_map is None:
            b.final_class = UNRESOLVED_CLASS
            b.merge_note = "residential; zone_map недоступна -> не удалось определить dense/sparse"
            continue

        zone_cls = zone_class_at(zone_map, zone_classes, cx, cy)
        if zone_cls in RESIDENTIAL_ZONE_CLASSES:
            b.final_class = zone_cls
            b.merge_note = f"residential; зона по центроиду = {zone_cls}"
        else:
            b.final_class = UNRESOLVED_CLASS
            b.merge_note = f"residential; зона по центроиду = {zone_cls} (не dense/sparse) -> нерешено"

    return buildings


def draw_merged_masks(
    image: Image.Image,
    buildings: list[Building],
    colors: dict[str, tuple[int, int, int]] = MERGED_COLORS,
    alpha: float = 0.55,
    outline_width: int = 2,
) -> Image.Image:
    """Визуализация итоговых 4(+1)-классовых масок зданий после слияния."""
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for b in buildings:
        if b.final_class is None:
            raise ValueError(
                f"Building {b.ann_id} не прошёл merge_zone_and_buildings (final_class is None)"
            )
        color = colors.get(b.final_class, (200, 200, 200))
        fill = (*color, int(255 * alpha))
        for poly in b.polygons:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 2:
                draw.polygon(pts, fill=fill, outline=(*color, 255), width=outline_width)

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def merge_summary(buildings: list[Building]) -> dict[str, int]:
    """Количество зданий по итоговому классу (после merge_zone_and_buildings)."""
    counts: dict[str, int] = {}
    for b in buildings:
        cls = b.final_class or "unresolved"
        counts[cls] = counts.get(cls, 0) + 1
    return counts
