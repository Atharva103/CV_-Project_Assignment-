import streamlit as st
import cv2
import numpy as np

# Streamlit setup
#st.title("Image Transformations with Affine Transformations")

st.markdown(
    """
    <style>
    body {
        background-image: linear-gradient(rgba(255, 165, 0, 0.5), rgba(0, 0, 128, 0.5));
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Image Transformations with Affine Transformations")
st.write("Name: Atharva Lokhande")
st.write("PRN: 202101040073")
st.write("Roll no: 035")


# Upload an image
st.header("Upload an image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Transformation selection
    st.header("Select Transformation")
    transformation_type = st.selectbox(
        "Choose a transformation:",
        ["Translation", "Rotation", "Scaling", "Shearing"]
    )

    if transformation_type == "Translation":
        st.subheader("Translation Parameters")
        tx = st.slider("X-axis Translation", -200, 200, 0)
        ty = st.slider("Y-axis Translation", -200, 200, 0)

        # Apply the translation transformation
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        transformed_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        st.image(transformed_image, caption="Transformed Image", use_column_width=True)

    elif transformation_type == "Rotation":
        st.subheader("Rotation Parameters")
        angle = st.slider("Angle (degrees)", -180, 180, 0)

        # Apply the rotation transformation
        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
        transformed_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        st.image(transformed_image, caption="Transformed Image", use_column_width=True)

    elif transformation_type == "Scaling":
        st.subheader("Scaling Parameters")
        scale_factor = st.slider("Scale Factor", 0.1, 2.0, 1.0)

        # Apply the scaling transformation
        transformed_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        st.image(transformed_image, caption="Transformed Image", use_column_width=True)

    elif transformation_type == "Shearing":
        st.subheader("Shearing Parameters")
        shear_x = st.slider("X-axis Shear", -2, 2, 0)
        shear_y = st.slider("Y-axis Shear", -2, 2, 0)

        # Apply the shearing transformation
        shear_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        transformed_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
        st.image(transformed_image, caption="Transformed Image", use_column_width=True)
