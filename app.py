import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import pathlib

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("Upload a driver image to detect drowsiness state and eye/yawn condition.")

# Load models
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model 1 - Drowsiness (EfficientNet-B0)
    model1 = models.efficientnet_b0(weights=None)
    in_features = model1.classifier[1].in_features
    model1.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    model1.load_state_dict(torch.load(
        'outputs/drowsiness_model.pth', 
        map_location=device))
    model1 = model1.to(device).eval()
    
    # Model 2 - Eye/Yawn (MobileNetV3)
    model2 = models.mobilenet_v3_small(weights=None)
    in_features = model2.classifier[3].in_features
    model2.classifier[3] = nn.Linear(in_features, 4)
    model2.load_state_dict(torch.load(
        'outputs/eye_yawn_model.pth',
        map_location=device))
    model2 = model2.to(device).eval()
    
    return model1, model2, device

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
**Two-stage detection pipeline:**
1. EfficientNet-B0 — Active vs Fatigue
2. MobileNetV3 — Eye/Yawn state

**Performance:**
- Drowsiness: 97.53% accuracy
- Eye/Yawn: 99.54% accuracy
""")

st.sidebar.header("EU Regulation")
st.sidebar.info("EU General Safety Regulation mandates driver monitoring in all new vehicles from 2024.")

# Main
uploaded_file = st.file_uploader(
    "Upload driver image",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with st.spinner("Analyzing..."):
        model1, model2, device = load_models()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            # Drowsiness prediction
            out1 = model1(img_tensor)
            prob1 = torch.softmax(out1, dim=1)[0]
            drowsy_classes = ['Active', 'Fatigue']
            drowsy_pred = drowsy_classes[out1.argmax().item()]
            drowsy_conf = prob1.max().item() * 100

            # Eye/Yawn prediction
            out2 = model2(img_tensor)
            prob2 = torch.softmax(out2, dim=1)[0]
            eye_classes = ['Closed', 'Open', 'No Yawn', 'Yawn']
            eye_pred = eye_classes[out2.argmax().item()]
            eye_conf = prob2.max().item() * 100

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Detection Results")

        # Drowsiness result
        if drowsy_pred == 'Fatigue':
            st.error(f"😴 DROWSY DRIVER DETECTED")
        else:
            st.success(f"✅ DRIVER ALERT")

        st.metric("Drowsiness State", drowsy_pred, 
                 f"{drowsy_conf:.1f}% confidence")

        st.divider()

        # Eye/Yawn result
        if eye_pred in ['Closed', 'Yawn']:
            st.warning(f"⚠️ {eye_pred.upper()} DETECTED")
        else:
            st.info(f"👁️ {eye_pred}")

        st.metric("Eye/Yawn State", eye_pred,
                 f"{eye_conf:.1f}% confidence")

        st.divider()

        # Confidence bars
        st.subheader("Confidence Scores")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Drowsiness**")
            for i, cls in enumerate(drowsy_classes):
                st.write(f"{cls}: {prob1[i].item()*100:.1f}%")
                st.progress(float(prob1[i].item()))
        
        with col4:
            st.markdown("**Eye/Yawn**")
            for i, cls in enumerate(eye_classes):
                st.write(f"{cls}: {prob2[i].item()*100:.1f}%")
                st.progress(float(prob2[i].item()))

else:
    st.info("👆 Upload a driver image to get started")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model 1: Drowsiness Detection")
        st.markdown("""
        - **Model:** EfficientNet-B0
        - **Classes:** Active, Fatigue
        - **Accuracy:** 97.53%
        - **Dataset:** 9,120 face images
        """)
    with col2:
        st.subheader("Model 2: Eye/Yawn Detection")
        st.markdown("""
        - **Model:** MobileNetV3-Small
        - **Classes:** Closed, Open, Yawn, No Yawn
        - **Accuracy:** 99.54%
        - **Dataset:** 2,900 images
        """)