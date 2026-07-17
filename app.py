"""Streamlit"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from merge_maps import MERGED_COLORS
from predict import load_pipeline, predict_building_crop, predict_tile_pipeline, tile_stage_images
from utils import BUILDING_COLORS, MODELS_DIR, ZONE_COLORS

st.set_page_config(page_title="UBC Building Pipeline", layout="wide")

ZONE_LABELS_RU = {
    "commercial": "commercial (коммерция)",
    "industrial": "industrial (промышленность)",
    "dense_residential": "dense_residential (плотная застройка)",
    "sparse_residential": "sparse_residential (редкая застройка)",
}
BUILDING_LABELS_RU = {
    "residential": "residential (жилое)",
    "commercial": "commercial (коммерция)",
    "industrial": "industrial (промышленность)",
}
MERGE_LABELS_RU = {**ZONE_LABELS_RU, "residential": "residential нерешённый (merge)"}


@st.cache_resource(show_spinner="Загрузка моделей…")
def get_pipeline():
    return load_pipeline()


def _prob_chart(probs: dict[str, float], title: str, labels: dict[str, str] | None = None) -> None:
    if not probs:
        st.warning("Нет данных для графика вероятностей.")
        return
    labels = labels or {}
    df = pd.DataFrame(
        {
            "class": [labels.get(k, k) for k in probs],
            "probability": list(probs.values()),
        }
    ).sort_values("probability", ascending=True)
    st.bar_chart(df.set_index("class"), height=280)
    st.caption(title)


def _render_legend(title: str, colors: dict[str, tuple[int, int, int]], labels: dict[str, str]) -> None:
    lines = []
    for key in colors:
        rgb = colors[key]
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        lines.append(f"<span style='color:{hex_color}'>■</span> {labels.get(key, key)}")
    st.markdown(f"**{title}**  \n" + "  \n".join(lines), unsafe_allow_html=True)


st.title("Классификация типа застройки — UBC pipeline")
st.markdown(
    "4 этапа на спутниковом снимке (**zone → find → class → merge**, ноутбук 07) "
    "или классификация **одного кропа здания** (требование `практика_ml.md`)."
)

with st.sidebar:
    st.header("Настройки")
    mode = st.radio(
        "Режим",
        options=("tile", "building"),
        format_func=lambda x: "Спутниковый снимок (4 этапа)" if x == "tile" else "Кроп одного здания",
    )
    uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png", "tif", "tiff", "webp"])
    st.divider()
    st.caption(f"Модели: `{MODELS_DIR}`")
    st.caption("Пороги: `reports/pipeline_calibrated_params.json`")

if uploaded is None:
    st.info("Загрузите изображение через боковую панель.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Загруженное изображение", use_container_width=True)

try:
    bundle = get_pipeline()
except FileNotFoundError as exc:
    st.error(f"Не найдены веса моделей: {exc}")
    st.stop()

with st.spinner("Inference…"):
    if mode == "building":
        result = predict_building_crop(image, bundle)
        st.subheader(f"Предсказанный класс: **{result['predicted_class']}**")
        st.metric("Уверенность", f"{result['confidence']:.1%}")
        _prob_chart(
            result["probabilities"],
            "Вероятности по классам здания (NY building classifier)",
            BUILDING_LABELS_RU,
        )
        st.dataframe(
            pd.DataFrame(
                [{"class": k, "probability": v} for k, v in result["probabilities"].items()]
            ).sort_values("probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        _render_legend("Легенда классов", BUILDING_COLORS, BUILDING_LABELS_RU)
        st.caption(f"Чекпоинт: `{bundle.class_ckpt.name}`")
    else:
        tile = predict_tile_pipeline(image, bundle)
        stages = tile_stage_images(tile)

        c1, c2, c3, c4 = st.columns(4)
        c1.image(stages["zone"], caption="1. Zone", use_container_width=True)
        c2.image(stages["find"], caption="2. Find", use_container_width=True)
        c3.image(stages["class"], caption="3. Class", use_container_width=True)
        c4.image(stages["merge"], caption="4. Merge", use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Зданий найдено", len(tile.buildings_find))
        m2.metric("Классифицировано", len(tile.buildings_class))
        m3.metric("Доминирующий merge-класс", tile.dominant_merged_class or "—")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Zone — средние вероятности по снимку")
            _prob_chart(tile.zone_probs_mean, "4 класса застройки (ConvNeXt zone)", ZONE_LABELS_RU)
        with col_b:
            st.subheader("Merge — доля зданий по итоговым классам")
            _prob_chart(tile.merge_distribution, "Итог после слияния zone + class", MERGE_LABELS_RU)

        with st.expander("Детали по зданиям"):
            rows = []
            for b in tile.buildings_merge:
                rows.append(
                    {
                        "pred_class": b.pred_class,
                        "final_class": b.final_class,
                        "confidence": b.confidence,
                        "merge_note": b.merge_note,
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write("Здания не найдены или отфильтрованы порогами.")

        with st.expander("Легенды цветов"):
            _render_legend("Zone", ZONE_COLORS, ZONE_LABELS_RU)
            _render_legend("Class", BUILDING_COLORS, BUILDING_LABELS_RU)
            _render_legend("Merge", MERGED_COLORS, MERGE_LABELS_RU)

        st.caption(
            f"zone: `{bundle.zone_ckpt.name}` | find: `{bundle.find_ckpt.name}` | "
            f"class: `{bundle.class_ckpt.name}` | пороги: {bundle.params.to_dict()}"
        )
