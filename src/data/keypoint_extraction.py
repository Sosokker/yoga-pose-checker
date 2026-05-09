import os
import cv2
import numpy as np
from typing import Optional
from tqdm import tqdm

# Try old API first, then new API
try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose
    _OLD_API = True
except AttributeError:
    _OLD_API = False
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    import mediapipe as mp


class MediaPipeKeypointExtractor:
    """
    Extracts 17 body keypoints from images using MediaPipe Pose.
    Each keypoint: (x, y, z, visibility)
    Compatible with both old (solutions) and new (tasks) MediaPipe APIs.
    """

    # MediaPipe Pose landmarks mapping to COCO-style 17 keypoints
    LANDMARK_INDICES = [
        0,   # nose
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
        23,  # left_hip
        24,  # right_hip
        25,  # left_knee
        26,  # right_knee
        27,  # left_ankle
        28,  # right_ankle
        5,   # left_eye (approx)
        2,   # right_eye (approx)
        7,   # left_ear (approx)
        8,   # right_ear (approx)
    ]

    KEYPOINT_NAMES = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle", "left_eye", "right_eye",
        "left_ear", "right_ear",
    ]

    def __init__(self, static_image_mode: bool = True, model_complexity: int = 1):
        if _OLD_API:
            self.pose = _mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._old_api = True
        else:
            # New Tasks API
            model_path = os.environ.get(
                "MEDIAPIPE_POSE_MODEL",
                "models/pose_landmarker_lite.task"
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"MediaPipe pose model not found at {model_path}. "
                    "Download it from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker "
                    "or set MEDIAPIPE_POSE_MODEL env var."
                )
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            self._old_api = False

    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract keypoints from a single image.
        Returns (17, 4) numpy array [x, y, z, visibility], or None if no person detected.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._old_api:
            results = self.pose.process(image_rgb)
            if results.pose_landmarks is None:
                return None
            landmarks = results.pose_landmarks.landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.landmarker.detect(mp_image)
            if not results.pose_landmarks:
                return None
            landmarks = results.pose_landmarks[0]  # first detected person

        keypoints = np.zeros((17, 4), dtype=np.float32)
        for i, idx in enumerate(self.LANDMARK_INDICES):
            landmark = landmarks[idx]
            keypoints[i, 0] = landmark.x
            keypoints[i, 1] = landmark.y
            keypoints[i, 2] = landmark.z
            keypoints[i, 3] = getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))

        return keypoints

    def extract_from_directory(
        self, image_dir: str, output_dir: str, label: Optional[str] = None
    ) -> dict:
        """
        Extract keypoints from all images in a directory and save as .npy files.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        extracted_count = 0
        failed_count = 0
        saved_paths = []

        for img_file in tqdm(image_files, desc=f"Extracting {label or os.path.basename(image_dir)}"):
            img_path = os.path.join(image_dir, img_file)
            try:
                keypoints = self.extract_from_image(img_path)
                if keypoints is not None:
                    base_name = os.path.splitext(img_file)[0]
                    save_path = os.path.join(output_dir, f"{base_name}.npy")
                    np.save(save_path, keypoints)
                    saved_paths.append(save_path)
                    extracted_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"  Warning: failed to extract {img_path}: {e}")
                failed_count += 1

        return {
            "extracted": extracted_count,
            "failed": failed_count,
            "total": len(image_files),
            "saved_paths": saved_paths,
            "label": label,
        }

    def __del__(self):
        if self._old_api and hasattr(self, "pose"):
            self.pose.close()
        elif not self._old_api and hasattr(self, "landmarker"):
            self.landmarker.close()


class KeypointAugmenter:
    """Apply augmentations to keypoint sequences."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_std: float = 0.01,
        scale_range: tuple = (0.9, 1.1),
        rotation_deg: float = 5.0,
    ):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotation_deg = rotation_deg

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a (17, 4) keypoint array.
        Args:
            keypoints: (17, 4) array [x, y, z, visibility]
        Returns:
            Augmented (17, 4) array
        """
        kp = keypoints.copy()

        # Random horizontal flip
        if np.random.rand() < self.flip_prob:
            kp = self._horizontal_flip(kp)

        # Gaussian noise on x, y
        noise = np.random.normal(0, self.noise_std, size=(17, 2))
        kp[:, :2] += noise

        # Random rotation around center (0.5, 0.5)
        if self.rotation_deg > 0:
            angle = np.random.uniform(-self.rotation_deg, self.rotation_deg)
            kp = self._rotate(kp, angle)

        # Random scaling
        scale = np.random.uniform(*self.scale_range)
        kp[:, :3] *= scale

        # Clip to valid ranges
        kp[:, 0] = np.clip(kp[:, 0], 0.0, 1.0)
        kp[:, 1] = np.clip(kp[:, 1], 0.0, 1.0)

        return kp

    def _horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        """Mirror the skeleton horizontally and swap left-right joints."""
        kp = keypoints.copy()
        kp[:, 0] = 1.0 - kp[:, 0]

        swap_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (9, 10), (11, 12), (13, 14), (15, 16),
        ]
        for left, right in swap_pairs:
            kp[[left, right]] = kp[[right, left]]

        return kp

    def _rotate(self, keypoints: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate keypoints around image center (0.5, 0.5)."""
        kp = keypoints.copy()
        cx, cy = 0.5, 0.5
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        x_rel = kp[:, 0] - cx
        y_rel = kp[:, 1] - cy

        kp[:, 0] = cx + x_rel * cos_a - y_rel * sin_a
        kp[:, 1] = cy + x_rel * sin_a + y_rel * cos_a

        return kp
