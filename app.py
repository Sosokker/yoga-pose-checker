import os
import sys

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.models.cross_modal_attention import CrossModalPoseClassifier
from src.data.keypoint_extraction import MediaPipeKeypointExtractor

# ── Skeleton connections (COCO-style 17 keypoints) ──
SKELETON_CONNECTIONS = [
    (0, 1),   # nose -> left_shoulder
    (0, 2),   # nose -> right_shoulder
    (1, 3),   # left_shoulder -> left_elbow
    (3, 5),   # left_elbow -> left_wrist
    (2, 4),   # right_shoulder -> right_elbow
    (4, 6),   # right_elbow -> right_wrist
    (1, 7),   # left_shoulder -> left_hip
    (2, 8),   # right_shoulder -> right_hip
    (7, 9),   # left_hip -> left_knee
    (9, 11),  # left_knee -> left_ankle
    (8, 10),  # right_hip -> right_knee
    (10, 12), # right_knee -> right_ankle
    (7, 8),   # left_hip -> right_hip
    (0, 13),  # nose -> left_eye
    (0, 14),  # nose -> right_eye
    (13, 15), # left_eye -> left_ear
    (14, 16), # right_eye -> right_ear
]

JOINT_NAMES = [
    "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee",
    "R_knee", "L_ankle", "R_ankle", "L_eye", "R_eye",
    "L_ear", "R_ear",
]

JOINT_COLORS = [
    (255, 0, 0),      # nose - red
    (0, 255, 0),      # L_shoulder - green
    (0, 255, 0),      # R_shoulder - green
    (0, 200, 0),      # L_elbow
    (0, 200, 0),      # R_elbow
    (0, 150, 0),      # L_wrist
    (0, 150, 0),      # R_wrist
    (255, 165, 0),    # L_hip - orange
    (255, 165, 0),    # R_hip - orange
    (200, 130, 0),    # L_knee
    (200, 130, 0),    # R_knee
    (150, 100, 0),    # L_ankle
    (150, 100, 0),    # R_ankle
    (0, 0, 255),      # L_eye - blue
    (0, 0, 255),      # R_eye - blue
    (0, 0, 200),      # L_ear
    (0, 0, 200),      # R_ear
]


def draw_skeleton(image: np.ndarray, keypoints: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw skeleton connections and joint circles on the image."""
    h, w = image.shape[:2]
    canvas = image.copy()

    # Draw connections
    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
        x1, y1, _, v1 = keypoints[start_idx]
        x2, y2, _, v2 = keypoints[end_idx]
        if v1 > 0.5 and v2 > 0.5:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(canvas, pt1, pt2, (255, 255, 255), thickness)

    # Draw joints
    for i, (kx, ky, _, vis) in enumerate(keypoints):
        if vis > 0.5:
            cx, cy = int(kx * w), int(ky * h)
            color = JOINT_COLORS[i]
            cv2.circle(canvas, (cx, cy), 6, color, -1)
            cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), 2)
            # Joint label
            cv2.putText(canvas, str(i), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(canvas, str(i), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return canvas


def create_keypoint_heatmap(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Create a heatmap-style visualization of joint confidence."""
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for kx, ky, _, vis in keypoints:
        if vis > 0.5:
            cx, cy = int(kx * w), int(ky * h)
            cv2.circle(heatmap, (cx, cy), 25, vis, -1)

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Colorize
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Overlay on original
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return overlay


# ── Page config ──
st.set_page_config(page_title="Yoga Pose Classifier", page_icon="🧘", layout="wide")

st.title("🧘 Yoga Pose Classification")
st.markdown("""
Classify yoga poses in real-time using **MediaPipe keypoint extraction** and a **Cross-Modal Attention** neural network.
Upload an image, capture from your camera, or drag-and-drop a photo.
""")

# ── Load model ──
@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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

    model = model.to(device)

    checkpoint_path = "outputs/checkpoints/cross_modal_full_bs64_best.pt"
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
    else:
        st.warning(f"No checkpoint found at {checkpoint_path}. Using random weights.")

    return model, classes, device


model, class_names, device = load_model()

# ── Sidebar ──
st.sidebar.header("🧠 Model Info")
st.sidebar.write(f"**Classes:** {len(class_names)}")
st.sidebar.write(f"**Device:** {device}")
st.sidebar.write(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
st.sidebar.write("---")
st.sidebar.write("**Supported Poses:**")
for c in class_names:
    st.sidebar.write(f"- {c}")
st.sidebar.write("---")
st.sidebar.write("**Pipeline:**")
st.sidebar.write("1. 📷 Capture image")
st.sidebar.write("2. 🦴 Extract 17 keypoints (MediaPipe)")
st.sidebar.write("3. 🔗 Draw skeleton")
st.sidebar.write("4. 🧠 Cross-modal attention")
st.sidebar.write("5. 📊 Predict pose class")

# ── Input selection ──
input_method = st.radio("Choose input method:", ["📁 Upload Image", "📷 Camera"], horizontal=True)

image = None
uploaded_file = None

camera_image = None
if input_method == "📷 Camera":
    st.info("Click the button below to capture from your webcam. Make sure you're in a well-lit area and your full body is visible.")
    camera_image = st.camera_input("Take a photo of your yoga pose")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
else:
    uploaded_file = st.file_uploader("Upload a yoga pose image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# ── Processing ──
if image is not None:
    # Convert to array
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Extract keypoints
    extractor = MediaPipeKeypointExtractor()
    temp_path = "/tmp/yoga_temp.jpg"
    cv2.imwrite(temp_path, img_bgr)
    keypoints = extractor.extract_from_image(temp_path)

    if keypoints is None:
        st.error("🚫 No person detected in the image. Please try another image with a clearer view of the person.")
    else:
        # Run inference
        x = torch.from_numpy(keypoints).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits, all_attn = model(x)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(torch.argmax(logits, dim=1).item())
        top3_idx = np.argsort(probs)[::-1][:3]

        # ── Show pipeline ──
        st.markdown("---")
        st.subheader("🔬 Processing Pipeline")

        # Step 1: Original
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Step 1: Original Image**")
            st.image(img_array, use_container_width=True)

        with col2:
            st.markdown("**Step 2: Skeleton Detection**")
            skeleton_img = draw_skeleton(img_array, keypoints, thickness=3)
            st.image(skeleton_img, use_container_width=True)

        with col3:
            st.markdown("**Step 3: Joint Confidence Heatmap**")
            heatmap_img = create_keypoint_heatmap(img_array, keypoints)
            st.image(heatmap_img, use_container_width=True)

        # ── Prediction results ──
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        pred_col, viz_col = st.columns([1, 2])

        with pred_col:
            st.markdown(f"### 🏆 Predicted: `{class_names[pred_class]}`")
            st.markdown(f"**Confidence:** {probs[pred_class]*100:.1f}%")

            st.markdown("#### Top-3 Predictions:")
            for rank, idx in enumerate(top3_idx, 1):
                prob = probs[idx]
                label = class_names[idx]
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                st.write(f"{emoji} **{label}** — {prob*100:.1f}%")
                st.progress(float(prob))

        with viz_col:
            st.markdown("**Cross-Modal Attention: Which joints matter for the prediction?**")
            attn = all_attn[0][0].cpu().numpy()
            attn_to_pred = attn[:, pred_class]

            fig, ax = plt.subplots(figsize=(12, 4))
            bars = ax.bar(range(17), attn_to_pred, color="steelblue", edgecolor="white", linewidth=0.5)
            top3_joints = np.argsort(attn_to_pred)[-3:]
            for j in top3_joints:
                bars[j].set_color("coral")
            ax.set_xticks(range(17))
            ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Attention Weight", fontsize=11)
            ax.set_title(f"Joint Attention for Predicted Class: {class_names[pred_class]}", fontsize=12)
            ax.set_ylim(0, max(attn_to_pred.max() * 1.2, 0.01))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            # Show all-class attention heatmap
            st.markdown("**Attention Heatmap: All joints × All classes**")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            im = ax2.imshow(attn, aspect="auto", cmap="viridis")
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha="right")
            ax2.set_yticks(range(17))
            ax2.set_yticklabels(JOINT_NAMES, fontsize=8)
            ax2.set_ylabel("Joint")
            ax2.set_xlabel("Class")
            ax2.set_title("Cross-Modal Attention Weights (Joint → Class)")
            plt.colorbar(im, ax=ax2, label="Attention Weight")
            plt.tight_layout()
            st.pyplot(fig2)

        # ── Keypoint details ──
        st.markdown("---")
        with st.expander("📋 Raw Keypoint Data"):
            kp_df = pd.DataFrame({
                "Joint": JOINT_NAMES,
                "X": keypoints[:, 0].round(4),
                "Y": keypoints[:, 1].round(4),
                "Z": keypoints[:, 2].round(4),
                "Visibility": keypoints[:, 3].round(4),
            })
            st.dataframe(kp_df, use_container_width=True)

            # Show joint coordinates as a simple scatter
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            x_coords = keypoints[:, 0]
            y_coords = 1 - keypoints[:, 1]  # Flip Y for visualization
            colors = [JOINT_COLORS[i] for i in range(17)]
            for i in range(17):
                if keypoints[i, 3] > 0.5:
                    ax3.scatter(x_coords[i], y_coords[i], c=[np.array(colors[i])/255], s=100, edgecolors="black")
                    ax3.annotate(str(i), (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
            # Draw connections
            for (start_idx, end_idx) in SKELETON_CONNECTIONS:
                if keypoints[start_idx, 3] > 0.5 and keypoints[end_idx, 3] > 0.5:
                    ax3.plot([x_coords[start_idx], x_coords[end_idx]],
                             [y_coords[start_idx], y_coords[end_idx]],
                             "w-", alpha=0.5, linewidth=1)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_aspect("equal")
            ax3.set_title("Pose Skeleton (normalized coordinates)")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y (inverted)")
            ax3.set_facecolor("black")
            fig3.patch.set_facecolor("black")
            st.pyplot(fig3)

        st.success("✅ Analysis complete!")

else:
    st.info("👆 Upload an image or capture from your camera to get started!")
