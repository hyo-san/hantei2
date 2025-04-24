import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="LV 真贋判定AI", layout="centered")
st.title("👜 Louis Vuitton 真贋判定AI")
st.write("画像をアップロードして、本物かどうかをチェックしてみましょう。")

# モデル読み込み（仮でランダム判定に）
def fake_model_predict(img_array):
    # ここでは仮にランダムな結果を返す（デモ用）
    confidence = np.random.uniform(0.6, 0.99)
    label = np.random.choice(["本物の可能性が高い", "偽物の可能性が高い"])
    return label, confidence

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="アップロードされた画像", use_column_width=True)

    # 画像処理
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # 仮判定
    with st.spinner("判定中..."):
        label, confidence = fake_model_predict(img_array)
        st.subheader(f"判定結果：{label}")
        st.write(f"信頼度：{confidence * 100:.2f}%")

    st.caption("※ 本AIはデモ用です。正式な真贋判定には専門家の鑑定をご利用ください。")
