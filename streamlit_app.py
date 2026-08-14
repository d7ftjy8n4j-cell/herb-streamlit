# -*- coding: utf-8 -*-
"""
中药材图像识别系统 - Streamlit 版
基于 PaddlePaddle ResNet-50 (PaddleX)，支持 163 种常见中药材识别。

本地运行:  streamlit run streamlit_app.py
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import streamlit as st
import cv2
from paddle.inference import Config, create_predictor

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_PATH = os.path.join(BASE_DIR, "model", "label.json")
PDMODEL_PATH = os.path.join(BASE_DIR, "model", "inference.pdmodel")
PDIPARAMS_PATH = os.path.join(BASE_DIR, "model", "inference.pdiparams")
CSV_PATH = os.path.join(BASE_DIR, "data", "t_medicine.csv")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")


def ensure_params() -> str:
    """若模型参数被拆分为多个 part（GitHub 上传限制），则合并为完整文件"""
    if os.path.exists(PDIPARAMS_PATH):
        return PDIPARAMS_PATH
    parts = sorted(glob.glob(os.path.join(BASE_DIR, "model", "inference.pdiparams.part*")))
    if len(parts) > 1:
        with open(PDIPARAMS_PATH, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    out.write(f.read())
        print(f"已合并 {len(parts)} 个模型分片 -> inference.pdiparams")
    return PDIPARAMS_PATH

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# 模型加载与推理
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="正在加载 ResNet-50 模型，首次约需 10 秒...")
def get_predictor():
    params_path = ensure_params()
    config = Config()
    config.set_model(PDMODEL_PATH, params_path)
    config.enable_memory_optim()
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(4)
    config.mkldnn_enabled()
    return create_predictor(config)


@st.cache_data
def load_label():
    with open(LABEL_PATH, encoding="utf-8") as f:
        return json.load(f)  # {"0": {"pinyin": ..., "name": ...}, ...}


@st.cache_data
def load_medicine_df():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    return df.set_index("chinese")


def preprocess(img_bytes: bytes) -> np.ndarray:
    """与 PaddleX 一致的预处理: resize 短边 -> 中心裁剪 224 -> 归一化"""
    img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解析图片，请上传 JPG/PNG 格式图片")

    percent = float(IMG_SIZE) / min(img.shape[0], img.shape[1])
    img = cv2.resize(
        img,
        (int(round(img.shape[1] * percent)), int(round(img.shape[0] * percent))),
    )
    h, w = img.shape[:2]
    ws, hs = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    img = img[hs : hs + IMG_SIZE, ws : ws + IMG_SIZE, :]

    img = img[:, :, ::-1].astype("float32").transpose((2, 0, 1)) / 255.0
    img = (img - MEAN) / STD
    return img[np.newaxis, :]


def predict(img_bytes: bytes) -> list:
    """返回 Top5: [{"name": 中文名, "pinyin": 拼音, "prob": 置信度%}, ...]"""
    predictor = get_predictor()
    img = preprocess(img_bytes)

    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        handle = predictor.get_input_handle(name)
        handle.reshape(img.shape)
        handle.copy_from_cpu(img)

    predictor.run()
    out = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()

    label = load_label()
    top_idx = np.argsort(out[0])[::-1][:5]
    results = []
    for idx in top_idx:
        info = label[str(idx)]
        results.append(
            {
                "name": info["name"],
                "pinyin": info["pinyin"],
                "prob": round(float(out[0][idx]) * 100, 2),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 页面 UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="中药材识别系统", page_icon="🌿", layout="wide")

st.title("🌿 中药材图像识别系统")
st.caption("基于 PaddlePaddle ResNet-50 · 支持 163 种常见中药材 · Top-1 准确率 98.1%")

sample_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
sample_names = {os.path.basename(p): p for p in sample_files}

tab_upload, tab_sample = st.tabs(["📤 上传图片", "🖼️ 示例图片"])

img_bytes = None

with tab_upload:
    up = st.file_uploader("选择一张中药材图片（JPG / PNG）", type=["jpg", "jpeg", "png"])
    if up is not None:
        img_bytes = up.getvalue()
        st.image(img_bytes, caption="你上传的图片", width=320)

with tab_sample:
    if sample_files:
        cols = st.columns(len(sample_files))
        chosen = None
        for col, path in zip(cols, sample_files):
            with col:
                st.image(path)
                if st.button("识别", key=path):
                    chosen = path
        if chosen is None:
            st.info("点击任意示例图下方的「识别」按钮进行体验")
        else:
            img_bytes = open(chosen, "rb").read()
            st.success(f"已选择示例图: {os.path.basename(chosen)}")

if img_bytes is not None:
    with st.spinner("🤖 识别中..."):
        results = predict(img_bytes)

    st.divider()
    st.subheader("📊 识别结果 (Top-5)")

    # 置信度条形图
    chart_df = pd.DataFrame(
        {"药材": [r["name"] for r in results][::-1],
         "置信度 %": [r["prob"] for r in results][::-1]}
    )
    st.bar_chart(chart_df.set_index("药材"), horizontal=True, color="#2e7d32")

    # Top-1 高亮
    top1 = results[0]
    st.markdown(
        f"### 🥇 识别结果: **{top1['name']}** "
        f"（置信度 {top1['prob']:.2f}%）"
    )

    # 药材百科信息
    df = load_medicine_df()
    if top1["name"] in df.index:
        m = df.loc[top1["name"]]
        st.markdown("#### 📖 药材百科")
        c1, c2, c3 = st.columns(3)
        c1.metric("拉丁名", m.get("latin", "未知") or "未知")
        c2.metric("科", m.get("family", "未知") or "未知")
        c3.metric("药用部位", m.get("use_part", "未知") or "未知")

        st.markdown(
            f"""
| 属性 | 内容 |
|------|------|
| **性** | {m.get('property', '未知') or '未知'} |
| **味** | {m.get('flavor', '未知') or '未知'} |
| **归经** | {m.get('meridian_tropism', '未知') or '未知'} |
| **功效分类** | {m.get('efficacy_class', '未知') or '未知'} |
| **产地** | {m.get('habitat', '未知') or '未知'} |
| **采集时间** | {m.get('collection_time', '未知') or '未知'} |
"""
        )
        st.markdown(f"**功效:** {m.get('indications', '未知') or '未知'}")
        st.markdown(f"**性状:** {m.get('appearance', '未知') or '未知'}")

    st.divider()
    st.markdown("**其他候选：** " + "、".join(
        f"{r['name']} ({r['prob']:.2f}%)" for r in results[1:]
    ))

st.divider()
st.caption(
    "⚠️ 识别结果仅供学习参考，不构成医疗建议。"
    "项目地址: github.com/mengze666/Image-recognition-of-Chinese-herbal-medicine"
)
