from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from jepa_inference import JEPAService


st.set_page_config(page_title="JEPA Mask Fill Demo", page_icon="🧩", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1d2b53 0%, #0b1020 42%, #050816 100%);
        color: #f5f7fb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stFileUploader"] {
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 0.5rem;
    }
    h1, h2, h3, p, label, span, div {
        color: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("JEPA mask-fill demo")
st.write("Upload an image, apply a random black square, and let the checkpoint guess the missing pixels.")

checkpoint_path = Path(__file__).with_name("mini_jepa_final.pt")

@st.cache_resource
def load_service() -> JEPAService:
    return JEPAService(checkpoint_path)


service = load_service()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1)

if uploaded_file is None:
    st.info("Choose a photo to see the masked and reconstructed result.")
    st.stop()

image = Image.open(uploaded_file)
result = service.reconstruct(image, seed=int(seed))

left, middle, right = st.columns(3)
with left:
    st.subheader("Original")
    st.image(result.original, use_container_width=True)
with middle:
    st.subheader("Masked")
    st.image(result.masked, use_container_width=True)
with right:
    st.subheader("Reconstructed")
    st.image(result.reconstructed, use_container_width=True)

st.caption(f"Masked region: {result.mask_box}")
