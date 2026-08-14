# 🌿 中药材图像识别系统（Streamlit 版）

基于 **PaddlePaddle ResNet-50**（PaddleX 训练）的中药材图像识别 Web 应用，支持 **163 种**常见中药材识别，Top-1 准确率 **98.1%**，Top-5 准确率 **99.8%**。

> 原项目: [mengze666/Image-recognition-of-Chinese-herbal-medicine](https://github.com/mengze666/Image-recognition-of-Chinese-herbal-medicine)（Flask 版）
> 本仓库是使用 Streamlit 重构的轻量版，开箱即用，支持部署到 Streamlit Community Cloud。

## ✨ 功能

- 📤 上传图片或点击示例图，一键识别中药材
- 📊 Top-5 识别结果 + 置信度条形图
- 📖 药材百科：拉丁名、科属、性味归经、功效、产地、性状等

## 🚀 本地运行

```bash
# 1. 创建虚拟环境 (Python 3.10 - 3.11)
conda create -n herb python=3.11 -y
conda activate herb

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动
streamlit run streamlit_app.py
```

浏览器访问 http://localhost:8501

## ☁️ 部署到 Streamlit Cloud（免费）

1. 把本仓库推送到你的 GitHub
2. 打开 https://share.streamlit.io （或 https://streamlit.io/cloud）→ **New app**
3. 选择本仓库，Main file 填 `streamlit_app.py`
4. 点击 Deploy，等待几分钟即可获得公网地址 `https://xxx.streamlit.app`

## 📁 项目结构

```
├── streamlit_app.py      # Streamlit 主程序
├── requirements.txt      # 依赖
├── model/                # ResNet-50 推理模型 (Paddle Inference)
│   ├── inference.pdmodel
│   ├── inference.pdiparams
│   └── label.json        # 163 类标签
├── data/
│   └── t_medicine.csv    # 163 种药材百科数据
└── sample_images/        # 示例图片（可直接体验）
```

## ⚠️ 说明

- 识别结果仅供学习参考，不构成医疗建议
- 数据集: https://aistudio.baidu.com/datasetdetail/246739（163 种中药材，训练集 25.7 万张）
