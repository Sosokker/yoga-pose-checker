import os
import sys

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.models.cross_modal_attention import CrossModalPoseClassifier
from src.data.keypoint_extraction import MediaPipeKeypointExtractor
from src.data.dataset import YogaPoseDataset

# Page config
st.set_page_config(page_title="Yoga Pose Classifier", page_icon="🧘", layout="wide")

st.title("🧘 Yoga Pose Classification")
st.markdown("""
Upload an image of a yoga pose and the model will classify it using **MediaPipe keypoint extraction**
combined with a **Cross-Modal Attention** neural network.
""")

# Load model
@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Determine number of classes from processed data
    data_dir = "data/processed/keypoints"
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    else:
        classes = ["downdog", "goddess", "plank", "tree", "warrior2"]
    num_classes = len(classes)

    model = CrossModalPoseClassifier(
        num_classes=num_classes,
        d_model=128,
        hidden_dim=128,
        num_attention_layers=2,
        dropout=0.2,
        embed_dropout=0.1,
        use_joint_self_attention=True,
        num_self_attention_layers=2,
        num_self_attention_heads=4,
        use_flatten_raw=True,
    )

    checkpoint_path = "outputs/checkpoints/cross_modal_full_bs64_best.pt"
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
    else:
        st.warning(f"No checkpoint found at {checkpoint_path}. Using random weights.")

    return model, classes, device

model, class_names, device = load_model()

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.write(f"Classes: {len(class_names)}")
st.sidebar.write(f"Device: {device}")
st.sidebar.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
st.sidebar.write("---")
st.sidebar.write("**Classes:**")
for c in class_names:
    st.sidebar.write(f"- {c}")

# Upload
uploaded_file = st.file_uploader("Upload a yoga pose image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        # Extract keypoints
        extractor = MediaPipeKeypointExtractor()
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Save temp and extract
        temp_path = "/tmp/yoga_temp.jpg"
        cv2.imwrite(temp_path, img_bgr)
        keypoints = extractor.extract_from_image(temp_path)

        if keypoints is None:
            st.error("No person detected in the image. Please try another image with a clearer view of the person.")
        else:
            # Run inference
            x = torch.from_numpy(keypoints).unsqueeze(0).float().to(device)  # (1, 17, 4)
            with torch.no_grad():
                logits, all_attn = model(x)

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            top3_idx = np.argsort(probs)[::-1][:3]

            # Show top-3 predictions
            for rank, idx in enumerate(top3_idx, 1):
                prob = probs[idx]
                label = class_names[idx]
                st.write(f"**#{rank} {label}** — {prob*100:.1f}%")
                st.progress(float(prob))

            # Show attention visualization
            st.subheader("Attention Visualization")
            attn = all_attn[0][0].cpu().numpy()  # first layer, first sample: (17, num_classes)
            pred_class = int(torch.argmax(logits, dim=1).item())
            attn_to_pred = attn[:, pred_class]

            joint_names = [
                "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
                "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee",
                "R_knee", "L_ankle", "R_ankle", "L_eye", "R_eye",
                "L_ear", "R_ear",
            ]

            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(17), attn_to_pred, color="steelblue")
            top3_joints = np.argsort(attn_to_pred)[-3:]
            for j in top3_joints:
                bars[j].set_color("coral")
            ax.set_xticks(range(17))
            ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Joint Attention for Predicted Class: {class_names[pred_class]}")
            ax.set_ylim(0, max(attn_to_pred.max() * 1.2, 0.01))
            plt.tight_layout()
            st.pyplot(fig)

            # Show keypoint overlay
            st.subheader("Detected Keypoints")
            h, w = img_array.shape[:2]
            overlay = img_array.copy()
            for i, (kx, ky, kz, vis) in enumerate(keypoints):
                if vis > 0.5:
                    cx, cy = int(kx * w), int(ky * h)
                    cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(overlay, str(i), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            st.image(overlay, use_container_width=True)

            # Show raw keypoint table
            with st.expander("Raw Keypoint Data"):
                kp_df = {
                    "Joint": joint_names,
                    "X": keypoints[:, 0].round(4),
                    "Y": keypoints[:, 1].round(4),
                    "Z": keypoints[:, 2].round(4),
                    "Visibility": keypoints[:, 3].round(4),
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(kp_df))
else:
    st.info("Upload an image to get started!")
