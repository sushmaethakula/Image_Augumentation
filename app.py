import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Image Transformer", layout="centered")
st.title("🖼️ Image Transformer")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        original_shape = img_array.shape

        # Show original in smaller size
        st.image(img_array, caption=f"Original Image | {original_shape[1]}x{original_shape[0]}", width=300)

        # Sidebar controls
        with st.sidebar.expander("🎨 Effects"):
            effect = st.selectbox("Choose Effect", ["None", "Grayscale", "Gaussian Blur", "Canny Edge Detection", "HSV", "LAB"])

        with st.sidebar.expander("🔄 Rotate"):
            rotate = st.slider("Rotate (degrees)", -180, 180, 0)

        with st.sidebar.expander("↔️ Flip"):
            flip_h = st.checkbox("Flip Horizontally")
            flip_v = st.checkbox("Flip Vertically")

        with st.sidebar.expander("📐 Shear & Translate"):
            shear_x = st.slider("Shear X", -0.5, 0.5, 0.0)
            shear_y = st.slider("Shear Y", -0.5, 0.5, 0.0)
            translate_x = st.slider("Translate X", -100, 100, 0)
            translate_y = st.slider("Translate Y", -100, 100, 0)

        with st.sidebar.expander("✂️ Crop"):
            crop = st.checkbox("Enable Cropping")
            if crop:
                max_top = original_shape[0] // 2
                max_side = original_shape[1] // 2
                crop_top = st.slider("Crop Top", 0, max_top, 0)
                crop_bottom = st.slider("Crop Bottom", 0, max_top, 0)
                crop_left = st.slider("Crop Left", 0, max_side, 0)
                crop_right = st.slider("Crop Right", 0, max_side, 0)

        # Start transformation
        transformed = img_array.copy()

        if effect == "Grayscale":
            transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
        elif effect == "Gaussian Blur":
            transformed = cv2.GaussianBlur(transformed, (11, 11), 0)
        elif effect == "Canny Edge Detection":
            gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
            transformed = cv2.Canny(gray, 100, 200)
        elif effect == "HSV":
            transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2HSV)
        elif effect == "LAB":
            transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2Lab)

        rows, cols = transformed.shape[:2]
        M = np.float32([[1, shear_x, translate_x], [shear_y, 1, translate_y]])
        transformed = cv2.warpAffine(transformed, M, (cols, rows))

        M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)
        transformed = cv2.warpAffine(transformed, M_rotate, (cols, rows))

        if flip_h:
            transformed = cv2.flip(transformed, 1)
        if flip_v:
            transformed = cv2.flip(transformed, 0)

        if crop:
            transformed = transformed[crop_top:rows - crop_bottom, crop_left:cols - crop_right]

        # Show transformed in smaller size
        st.image(transformed, caption="Transformed Image", width=300, channels="GRAY" if len(transformed.shape) == 2 else "RGB")

        # Download button
        final_image = Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_GRAY2RGB) if len(transformed.shape) == 2 else transformed)
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("📥 Download Transformed Image", byte_im, "transformed.png", "image/png")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("📌 Upload an image to get started.")
