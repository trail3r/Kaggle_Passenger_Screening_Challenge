import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ultralytics import YOLO

from src.dataset import get_2d_views

VIEW = 16

HEIGHT = 660
WIDTH = 512
MODEL_HEIGHT = 661

KEYPOINT_COUNT = 17
ZONE_COUNT = 17

ARTIFACT_VERSION = "pose_roi_native_v1"
SCHEMA_VERSION = 2
PIPELINE_REVISION = 2

MINIMUM_LOCKED_POSE_SCANS = 10
MINIMUM_LOCKED_POSE_POINTS = 800
MINIMUM_LOCKED_ROI_SCANS = 5
MINIMUM_LOCKED_ROIS = 200


# YOLOv8 Pose COCO 관절 포인트
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# 논문과 실제 ROI 생성에 사용하는 12개 관절만 GT 평가 대상으로 삼습니다.
POSE_JOINTS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

KEYPOINT_NAMES = {
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_WRIST: "left_wrist",
    RIGHT_WRIST: "right_wrist",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
    LEFT_KNEE: "left_knee",
    RIGHT_KNEE: "right_knee",
    LEFT_ANKLE: "left_ankle",
    RIGHT_ANKLE: "right_ankle",
}

POSE_JOINT_GROUPS = {
    LEFT_SHOULDER: "shoulder_hip",
    RIGHT_SHOULDER: "shoulder_hip",
    LEFT_HIP: "shoulder_hip",
    RIGHT_HIP: "shoulder_hip",
    LEFT_ELBOW: "elbow_knee",
    RIGHT_ELBOW: "elbow_knee",
    LEFT_KNEE: "elbow_knee",
    RIGHT_KNEE: "elbow_knee",
    LEFT_WRIST: "wrist_ankle",
    RIGHT_WRIST: "wrist_ankle",
    LEFT_ANKLE: "wrist_ankle",
    RIGHT_ANKLE: "wrist_ankle",
}

ROI_ZONE_GROUPS = {
    1: "arm",
    2: "arm",
    3: "arm",
    4: "arm",
    5: "torso",
    6: "torso",
    7: "torso",
    8: "thigh",
    9: "torso",
    10: "thigh",
    11: "thigh",
    12: "thigh",
    13: "calf_ankle",
    14: "calf_ankle",
    15: "calf_ankle",
    16: "calf_ankle",
    17: "torso",
}

# 관절 - 관절 사이의 범위 정의
# (시작 관절, 끝 관절, 시작 비율, 끝 비율)
JOINTS = {
    1: (RIGHT_SHOULDER, RIGHT_ELBOW, 0.00, 1.00),
    2: (RIGHT_ELBOW, RIGHT_WRIST, 0.00, 1.00),
    3: (LEFT_SHOULDER, LEFT_ELBOW, 0.00, 1.00),
    4: (LEFT_ELBOW, LEFT_WRIST, 0.00, 1.00),
    8: (RIGHT_HIP, RIGHT_KNEE, 0.00, 0.58),
    10: (LEFT_HIP, LEFT_KNEE, 0.00, 0.58),
    11: (RIGHT_HIP, RIGHT_KNEE, 0.42, 1.00),
    12: (LEFT_HIP, LEFT_KNEE, 0.42, 1.00),
    13: (RIGHT_KNEE, RIGHT_ANKLE, 0.00, 1.00),
    14: (LEFT_KNEE, LEFT_ANKLE, 0.00, 1.00),
    15: (RIGHT_KNEE, RIGHT_ANKLE, 0.82, 1.12),
    16: (LEFT_KNEE, LEFT_ANKLE, 0.82, 1.12),
}

LOWER_BODY_JOINTS = [
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
]

SYMMETRIC_JOINTS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
]

STRUCTURAL_JOINTS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

ARM_CHAINS = [
    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
]

# 좌표의 출처를 저장하여 관측값과 추론값을 구분합니다.
KEYPOINT_INVALID = 0
KEYPOINT_OBSERVED = 1
KEYPOINT_SWAPPED = 2
KEYPOINT_ROTATION = 3
KEYPOINT_ARM = 4
KEYPOINT_INTERPOLATED = 5

# 신체 구역의 생성 방법을 저장합니다.
ROI_INVALID = 0
ROI_INTENSITY = 1
ROI_ORIENTED = 2
ROI_TORSO = 3
ROI_RELATIVE = 4

LOWER_BODY_CORRECTION = 1
ROTATION_CORRECTION = 2
STRUCTURAL_REPLACEMENT = 4
ARM_REPLACEMENT = 8
INTERPOLATION = 16
ARM_REJECTION = 32

ROTATION_JOINTS = {
    "Shoulder": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "Hip": (LEFT_HIP, RIGHT_HIP),
    "Knee": (LEFT_KNEE, RIGHT_KNEE),
    "Ankle": (LEFT_ANKLE, RIGHT_ANKLE),
}

# 측면 View에서는 좌우 관절이 거의 겹치므로 간격이 충분한 경우에만 좌우를 판정합니다.
MINIMUM_GAP = 15
MINIMUM_CONFIDENCE = 0.5
RELIABLE_CONFIDENCE = 0.7
MINIMUM_POSE_CONFIDENCE = 0.25

VIEW_ANGLES = np.arange(VIEW) * 2 * np.pi / VIEW
ROTATION_MATRIX = np.stack([np.ones(VIEW), np.cos(VIEW_ANGLES), np.sin(VIEW_ANGLES)], axis=1)

HUBER_DELTA = 15
ROTATION_FIT_ITERATIONS = 5
ROTATION_PAIR_SWAP_RATIO = 0.70
ROTATION_TOTAL_SWAP_RATIO = 0.65
ROTATION_MINIMUM_VOTES = 3
ROTATION_MINIMUM_SUPPORT = 8
ROTATION_MAXIMUM_GAP = 3
ROTATION_MAXIMUM_SWAPS = 4

SIDE_VIEWS = {4, 5, 11, 12}

# 논문의 상세 설정이 공개되지 않아 native 해상도에 맞춘 초기값을 사용합니다.
LIMB_WIDTH = {
    1: (0.38, 24, 72),
    2: (0.32, 22, 64),
    3: (0.38, 24, 72),
    4: (0.32, 22, 64),
    8: (0.65, 36, 96),
    10: (0.65, 36, 96),
    11: (0.60, 32, 88),
    12: (0.60, 32, 88),
    13: (0.38, 26, 72),
    14: (0.38, 26, 72),
    15: (0.85, 24, 64),
    16: (0.85, 24, 64),
}

LIMB_GAUSSIAN_KERNEL = (9, 9)
LIMB_GAUSSIAN_SIGMA = 2.0
LIMB_POLYNOMIAL_DEGREE = 6
LIMB_SEARCH_SCALE = 1.25
LIMB_MINIMUM_LENGTH = 12
LIMB_MINIMUM_CONTRAST = 0.05

TORSO_PRIMARY_VIEWS = {
    5: {15, 0, 1},
    17: {7, 8, 9},
}
TORSO_SUPPORT_VIEWS = {
    5: {14, 2},
    17: {6, 10},
}
TORSO_MINIMUM_AREA = 200
TORSO_MINIMUM_EDGE = 10

# Pose를 사용할 수 없을 때만 적용하는 신체 Bounding Box 기준값입니다.
ZONE_ANCHORS = {
    1: (0.10, 0.34, 0.50, "right"),
    2: (0.00, 0.21, 0.48, "right"),
    3: (0.10, 0.34, 0.50, "left"),
    4: (0.00, 0.21, 0.48, "left"),
    5: (0.22, 0.47, 0.74, "center"),
    6: (0.38, 0.62, 0.62, "right"),
    7: (0.38, 0.62, 0.62, "left"),
    8: (0.57, 0.74, 0.52, "right"),
    9: (0.55, 0.71, 0.70, "center"),
    10: (0.57, 0.74, 0.52, "left"),
    11: (0.69, 0.84, 0.50, "right"),
    12: (0.69, 0.84, 0.50, "left"),
    13: (0.81, 0.96, 0.46, "right"),
    14: (0.81, 0.96, 0.46, "left"),
    15: (0.92, 1.00, 0.48, "right"),
    16: (0.92, 1.00, 0.48, "left"),
    17: (0.20, 0.46, 0.76, "center"),
}

SKELETON = [
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]

# Contrast Limited Adaptive Histogram Equalization
CLAHE_LIMIT = 3.0
CLAHE_GRID = (8, 8)

BODY_MASK_PERCENTILE = 75


def compensator(start, end, ratio):
    """관절 사이에서 지정된 비율에 해당하는 좌표를 계산합니다."""
    return start + (end - start) * ratio


def segmentor(keypoints, zone):
    """관절 좌표를 통해 신체 구역의 시작점과 끝점을 계산합니다."""
    start_keypoint, end_keypoint, start_ratio, end_ratio = JOINTS[zone]

    start = keypoints[start_keypoint]
    end = keypoints[end_keypoint]

    start_point = compensator(start, end, start_ratio)
    end_point = compensator(start, end, end_ratio)

    return start_point, end_point


def clip_box(box):
    """Bounding Box를 native 이미지의 half-open 좌표로 제한합니다."""
    x1, y1, x2, y2 = box

    x1 = int(np.clip(np.floor(x1), 0, WIDTH - 1))
    y1 = int(np.clip(np.floor(y1), 0, HEIGHT - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, WIDTH))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, HEIGHT))

    return np.array([x1, y1, x2, y2], dtype=np.int16)


def valid_point(point):
    """관절 좌표가 native 이미지 안에 있는지 확인합니다."""
    return bool(np.isfinite(point).all() and 0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT)


def circular_maximum_gap(indices):
    """관측된 View 사이의 가장 긴 원형 공백을 계산합니다."""
    indices = sorted(int(index) for index in indices)

    if not indices:
        return VIEW

    gaps = [end - start - 1 for start, end in zip(indices, indices[1:])]
    gaps.append(VIEW - indices[-1] - 1 + indices[0])

    return max(gaps)


def circular_maximum_run(values):
    """연속된 True 값의 가장 긴 원형 구간을 계산합니다."""
    values = np.asarray(values, dtype=bool)

    if values.all():
        return VIEW

    doubled = np.concatenate([values, values])
    longest = 0
    current = 0

    for value in doubled:
        current = current + 1 if value else 0
        longest = max(longest, current)

    return min(longest, VIEW)


def points_in_mask(points, mask, radius=0):
    """관절 좌표가 느슨한 신체 마스크 안에 있는지 확인합니다."""
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0

    result = []

    for point in points:
        if not valid_point(point):
            result.append(False)
            continue

        x, y = np.round(point).astype(int)
        x = min(x, WIDTH - 1)
        y = min(y, HEIGHT - 1)
        result.append(bool(mask[y, x]))

    return np.array(result, dtype=bool)


def converter(image):
    """`_.aps` 이미지를 Pose Estimator의 입력 이미지로 변환합니다."""
    image = image.astype(np.float32)

    # 반사 영향을 제한하고 CLAHE로 대비를 높여 신체 윤곽을 선명하게 만듭니다.
    maximum = np.percentile(image, 99.5)
    if maximum <= 0:
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    image = np.clip(image / maximum, 0, 1)
    image = (image * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=CLAHE_GRID)
    image = clahe.apply(image)

    # 1채널을 3채널로 복제하여 RGB 이미지로 학습된 Pose Estimator의 입력 형식에 맞춥니다.
    image = np.stack([image, image, image], axis=-1)

    return image


def decomposer(image):
    """밀리미터파 스캔 이미지에서 관절의 신체 영역 포함 여부를 확인할 느슨한 마스크를 생성합니다."""
    # 반사 강도가 낮은 부위도 포함하기 위해 상위 25% 픽셀을 후보 영역으로 선정합니다.
    threshold = np.percentile(image, BODY_MASK_PERCENTILE)
    mask = image > threshold

    return mask


def adapter(infile):
    """`_.aps` 파일을 Pose Estimator의 입력 데이터로 변환합니다."""
    views = get_2d_views(infile)

    images = []
    masks = []

    for image in views:
        images.append(converter(image))
        masks.append(decomposer(image))

    images = np.stack(images)
    masks = np.stack(masks)

    return images, masks


def detector(model, images):
    """16개 이미지 View에서 신체 관절 좌표와 신뢰도를 추출합니다."""

    # 검출에 실패한 이미지 View는 0으로 설정하여 이후 이미지 정제 단계에서 제외합니다.
    keypoints = np.zeros((VIEW, KEYPOINT_COUNT, 2), dtype=np.float32)
    keypoint_confidence = np.zeros((VIEW, KEYPOINT_COUNT), dtype=np.float32)
    frame_confidence = np.zeros(VIEW, dtype=np.float32)
    body_boxes = np.full((VIEW, 4), -1, dtype=np.int16)

    # 16개의 View를 한 번에 추론하고, 각 View에서 가장 신뢰도가 높은 사람을 한 명 선택합니다.
    results = model(list(images), verbose=False)

    for view, result in enumerate(results):
        if len(result.boxes) == 0 or result.keypoints is None:
            continue

        confidence = result.boxes.conf.cpu().numpy()
        person = int(confidence.argmax())

        detected_keypoints = result.keypoints.xy[person].cpu().numpy()
        detected_confidence = result.keypoints.conf[person].cpu().numpy()

        if len(detected_keypoints) == KEYPOINT_COUNT:
            keypoints[view] = detected_keypoints
            keypoint_confidence[view] = detected_confidence
        elif len(detected_keypoints) == len(POSE_JOINTS):
            keypoints[view, POSE_JOINTS] = detected_keypoints
            keypoint_confidence[view, POSE_JOINTS] = detected_confidence
        else:
            raise ValueError(f"Pose model returned {len(detected_keypoints)} keypoints")

        frame_confidence[view] = confidence[person]

        x1, y1, x2, y2 = result.boxes.xyxy[person].cpu().numpy()
        body_boxes[view] = clip_box((x1, y1, x2, y2))

    return keypoints, keypoint_confidence, frame_confidence, body_boxes


def align_lower_body(keypoints, keypoint_confidence):
    """골반의 좌우 순서를 기준으로 무릎과 발목 관절을 좌우 정렬합니다."""

    # 원본은 건들면 안 됩니다!
    aligned_keypoints = keypoints.copy()
    aligned_confidence = keypoint_confidence.copy()

    corrected = set()

    for view in range(VIEW):
        hip_confidence = aligned_confidence[view, [LEFT_HIP, RIGHT_HIP]]

        # 신뢰도가 낮은 골반을 기준으로 하체를 교정하지 않습니다.
        if hip_confidence.min() < MINIMUM_CONFIDENCE:
            continue

        hip_gap = aligned_keypoints[view, LEFT_HIP, 0] - aligned_keypoints[view, RIGHT_HIP, 0]

        # 골반의 좌우가 겹치는 경우에는 잘못된 교정을 피하기 위해 생략합니다.
        if abs(hip_gap) < MINIMUM_GAP:
            continue

        for left, right in LOWER_BODY_JOINTS:
            joint_confidence = aligned_confidence[view, [left, right]]

            # 신뢰도가 낮은 관절은 다음 단계에서 처리하기 위해 교정하지 않습니다.
            if joint_confidence.min() < MINIMUM_CONFIDENCE:
                continue

            joint_gap = aligned_keypoints[view, left, 0] - aligned_keypoints[view, right, 0]

            if abs(joint_gap) < MINIMUM_GAP:
                continue

            # 하체의 좌우 순서가 같은지 확인합니다.
            if np.sign(hip_gap) == np.sign(joint_gap):
                continue

            # Swap!
            aligned_keypoints[view, [left, right]] = aligned_keypoints[view, [right, left]]
            aligned_confidence[view, [left, right]] = aligned_confidence[view, [right, left]]

            corrected.add(view)

    return aligned_keypoints, aligned_confidence, sorted(corrected)


def fit_rotation_curve(coordinate, confidence, minimum_support=ROTATION_MINIMUM_SUPPORT):
    """관절 하나의 좌표에 신뢰도를 가중한 16-View 회전 곡선을 맞춥니다."""
    coordinate = np.asarray(coordinate, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)

    valid = np.isfinite(coordinate) & np.isfinite(confidence)
    valid &= confidence >= MINIMUM_CONFIDENCE
    support = np.where(valid)[0]

    if len(support) < minimum_support:
        return np.full(VIEW, np.nan, dtype=np.float32)

    if circular_maximum_gap(support) > ROTATION_MAXIMUM_GAP:
        return np.full(VIEW, np.nan, dtype=np.float32)

    weight = np.where(valid, confidence, 0)
    prediction = np.full(VIEW, np.nan, dtype=np.float64)

    for _ in range(ROTATION_FIT_ITERATIONS):
        root_weight = np.sqrt(weight)
        weighted_matrix = ROTATION_MATRIX * root_weight[:, None]
        weighted_coordinate = np.where(valid, coordinate, 0) * root_weight

        if np.linalg.matrix_rank(weighted_matrix) < 3:
            return np.full(VIEW, np.nan, dtype=np.float32)

        parameter = np.linalg.lstsq(weighted_matrix, weighted_coordinate, rcond=None)[0]
        prediction = ROTATION_MATRIX @ parameter

        residual = np.abs(np.where(valid, coordinate, prediction) - prediction)
        huber_weight = np.minimum(1, HUBER_DELTA / np.maximum(residual, 1e-6))
        weight = np.where(valid, confidence * huber_weight, 0)

    return prediction.astype(np.float32)


def predict_rotation(keypoints, keypoint_confidence, minimum_support=ROTATION_MINIMUM_SUPPORT):
    """17개 관절이 회전하는 2차원 궤도를 계산합니다."""
    prediction = np.full((VIEW, KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)

    for joint in range(KEYPOINT_COUNT):
        confidence = keypoint_confidence[:, joint].copy()
        inside = np.array([valid_point(point) for point in keypoints[:, joint]])
        confidence[~inside] = 0

        for coordinate in range(2):
            prediction[:, joint, coordinate] = fit_rotation_curve(
                keypoints[:, joint, coordinate],
                confidence,
                minimum_support,
            )

    return prediction


def huber_loss(residual):
    """작은 오차는 제곱하고 큰 오차의 영향은 선형으로 제한합니다."""
    absolute = abs(residual)

    if absolute <= HUBER_DELTA:
        return 0.5 * absolute**2

    return HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA)


def find_rotation_swaps(keypoints, keypoint_confidence, prediction):
    """여러 관절 쌍의 회전 궤적을 기준으로 좌우 반전 후보 View를 찾습니다."""
    candidates = []

    for view in range(VIEW):
        valid_pairs = 0
        swap_votes = 0
        hip_vote = False

        current_error = 0
        swapped_error = 0

        for left, right in ROTATION_JOINTS.values():
            left_confidence = keypoint_confidence[view, left]
            right_confidence = keypoint_confidence[view, right]

            if min(left_confidence, right_confidence) < MINIMUM_CONFIDENCE:
                continue

            left_x = keypoints[view, left, 0]
            right_x = keypoints[view, right, 0]

            predicted_left = prediction[view, left, 0]
            predicted_right = prediction[view, right, 0]

            if not np.isfinite([predicted_left, predicted_right]).all():
                continue

            # 측면 View처럼 실제 또는 예측 좌우가 겹치는 경우에는 판정하지 않습니다.
            if abs(left_x - right_x) < MINIMUM_GAP:
                continue

            if abs(predicted_left - predicted_right) < MINIMUM_GAP:
                continue

            pair_current_error = left_confidence * huber_loss(left_x - predicted_left)
            pair_current_error += right_confidence * huber_loss(right_x - predicted_right)

            # 좌표를 교환할 경우 해당 좌표의 신뢰도도 함께 이동한다고 가정합니다.
            pair_swapped_error = right_confidence * huber_loss(right_x - predicted_left)
            pair_swapped_error += left_confidence * huber_loss(left_x - predicted_right)

            valid_pairs += 1
            current_error += pair_current_error
            swapped_error += pair_swapped_error

            if pair_swapped_error < pair_current_error * ROTATION_PAIR_SWAP_RATIO:
                swap_votes += 1

                if left == LEFT_HIP:
                    hip_vote = True

        # 네 관절 쌍 중 최소 세 쌍이 동의할 때만 몸 전체의 반전 후보로 인정합니다.
        if valid_pairs < ROTATION_MINIMUM_VOTES:
            continue

        if swap_votes < ROTATION_MINIMUM_VOTES:
            continue

        if not hip_vote:
            continue

        if swapped_error >= current_error * ROTATION_TOTAL_SWAP_RATIO:
            continue

        candidates.append(view)

    return candidates


def rotation_objective(keypoints, keypoint_confidence):
    """구조 관절의 회전 궤적 오차를 계산합니다."""
    prediction = predict_rotation(keypoints, keypoint_confidence)
    residuals = []

    for left, right in ROTATION_JOINTS.values():
        for joint in (left, right):
            valid = keypoint_confidence[:, joint] >= MINIMUM_CONFIDENCE
            valid &= np.isfinite(prediction[:, joint, 0])
            valid &= np.array([valid_point(point) for point in keypoints[:, joint]])

            if valid.any():
                residuals.extend(np.abs(keypoints[valid, joint, 0] - prediction[valid, joint, 0]))

    if not residuals:
        return np.inf, np.inf

    residuals = np.asarray(residuals)
    return float(residuals.mean()), float(np.percentile(residuals, 90))


def anchor_violation_count(keypoints, keypoint_confidence):
    """정면과 후면에서 anatomical 좌우 순서가 어긋난 횟수를 계산합니다."""
    violations = 0

    for view, expected_sign in [(15, 1), (0, 1), (1, 1), (7, -1), (8, -1), (9, -1)]:
        for left, right in ROTATION_JOINTS.values():
            confidence = keypoint_confidence[view, [left, right]]

            if confidence.min() < MINIMUM_CONFIDENCE:
                continue

            gap = keypoints[view, left, 0] - keypoints[view, right, 0]

            if abs(gap) >= MINIMUM_GAP and np.sign(gap) != expected_sign:
                violations += 1

    return violations


def anchor_supports_swap(view, keypoints, keypoint_confidence):
    """정면·후면 View에서는 해부학적 좌우 순서가 개선될 때만 반전을 허용합니다."""
    if view in {15, 0, 1}:
        expected_sign = 1
    elif view in {7, 8, 9}:
        expected_sign = -1
    else:
        return True

    correct = 0
    incorrect = 0

    for left, right in ROTATION_JOINTS.values():
        confidence = keypoint_confidence[view, [left, right]]

        if confidence.min() < MINIMUM_CONFIDENCE:
            continue

        gap = keypoints[view, left, 0] - keypoints[view, right, 0]

        if abs(gap) < MINIMUM_GAP:
            continue

        if np.sign(gap) == expected_sign:
            correct += 1
        else:
            incorrect += 1

    return incorrect >= ROTATION_MINIMUM_VOTES and incorrect > correct


def arm_assignment_cost(keypoints, confidence, prediction, view, swap=False):
    """어깨와 팔 사슬의 연결 및 회전 궤적 비용을 계산합니다."""
    if prediction.shape != (VIEW, KEYPOINT_COUNT, 2):
        raise ValueError("prediction must have shape (16, 17, 2)")

    if swap:
        pairs = [
            (LEFT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST),
        ]
    else:
        pairs = [
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_ELBOW, RIGHT_WRIST),
        ]

    cost = 0.0

    for shoulder, elbow, wrist, target_elbow, target_wrist in pairs:
        joints = [shoulder, elbow, wrist]

        if confidence[view, joints].min() < MINIMUM_POSE_CONFIDENCE:
            return np.inf

        if not all(valid_point(keypoints[view, joint]) for joint in joints):
            return np.inf

        if not np.isfinite(prediction[view, [target_elbow, target_wrist]]).all():
            return np.inf

        cost += np.linalg.norm(keypoints[view, shoulder] - keypoints[view, elbow])
        cost += np.linalg.norm(keypoints[view, elbow] - keypoints[view, wrist])

        cost += 0.25 * np.linalg.norm(keypoints[view, elbow] - prediction[view, target_elbow])
        cost += 0.25 * np.linalg.norm(keypoints[view, wrist] - prediction[view, target_wrist])

    return float(cost)


def apply_rotation_swaps(keypoints, confidence, source, origin_joint, candidates, prediction):
    """확정된 View의 구조 관절과 일관된 팔 사슬을 좌우 교환합니다."""
    corrected_keypoints = keypoints.copy()
    corrected_confidence = confidence.copy()
    corrected_source = source.copy()
    corrected_origin = origin_joint.copy()
    applied = []

    structural_pairs = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_KNEE, RIGHT_KNEE),
        (LEFT_ANKLE, RIGHT_ANKLE),
    ]

    for view in candidates:
        if not anchor_supports_swap(view, corrected_keypoints, corrected_confidence):
            continue

        for left, right in structural_pairs:
            pair_confidence = corrected_confidence[view, [left, right]]

            if pair_confidence.min() < MINIMUM_CONFIDENCE:
                continue

            if not valid_point(corrected_keypoints[view, left]):
                continue

            if not valid_point(corrected_keypoints[view, right]):
                continue

            corrected_keypoints[view, [left, right]] = corrected_keypoints[view, [right, left]]
            corrected_confidence[view, [left, right]] = corrected_confidence[view, [right, left]]
            corrected_origin[view, [left, right]] = corrected_origin[view, [right, left]]
            corrected_source[view, [left, right]] = KEYPOINT_SWAPPED

        current_arm_cost = arm_assignment_cost(corrected_keypoints, corrected_confidence, prediction, view)
        swapped_arm_cost = arm_assignment_cost(
            corrected_keypoints,
            corrected_confidence,
            prediction,
            view,
            swap=True,
        )

        if swapped_arm_cost < current_arm_cost * 0.75:
            for left, right in [(LEFT_ELBOW, RIGHT_ELBOW), (LEFT_WRIST, RIGHT_WRIST)]:
                corrected_keypoints[view, [left, right]] = corrected_keypoints[view, [right, left]]
                corrected_confidence[view, [left, right]] = corrected_confidence[view, [right, left]]
                corrected_origin[view, [left, right]] = corrected_origin[view, [right, left]]
                corrected_source[view, [left, right]] = KEYPOINT_SWAPPED

        applied.append(view)

    return corrected_keypoints, corrected_confidence, corrected_source, corrected_origin, applied


def estimate_body_height(keypoints, keypoint_confidence):
    """정상 관측 View에서 어깨와 발목 사이의 신체 높이를 추정합니다."""
    heights = []

    for view in range(VIEW):
        joints = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE]

        if keypoint_confidence[view, joints].min() < MINIMUM_CONFIDENCE:
            continue

        shoulder = keypoints[view, [LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
        ankle = keypoints[view, [LEFT_ANKLE, RIGHT_ANKLE]].mean(axis=0)
        heights.append(np.linalg.norm(shoulder - ankle))

    return float(np.median(heights)) if heights else HEIGHT * 0.72


def reliable_neighbors(view, reliable):
    """대상 View의 양쪽 두 칸 안에 믿을 관측이 있는지 확인합니다."""
    left = any(reliable[(view - distance) % VIEW] for distance in (1, 2))
    right = any(reliable[(view + distance) % VIEW] for distance in (1, 2))
    return left and right


def replace_structural_outliers(keypoints, confidence, source, origin_joint, masks):
    """회전 궤도에서 명백히 벗어난 구조 관절만 선택적으로 대체합니다."""
    refined = keypoints.copy()
    refined_confidence = confidence.copy()
    refined_source = source.copy()
    refined_origin = origin_joint.copy()
    replaced_views = set()

    prediction = predict_rotation(keypoints, confidence)
    body_height = estimate_body_height(keypoints, confidence)

    for joint in STRUCTURAL_JOINTS:
        if not np.isfinite(prediction[:, joint]).all():
            continue

        residual = np.linalg.norm(keypoints[:, joint] - prediction[:, joint], axis=1)
        reliable = confidence[:, joint] >= MINIMUM_CONFIDENCE
        reliable &= np.array([valid_point(point) for point in keypoints[:, joint]])

        if reliable.sum() < ROTATION_MINIMUM_SUPPORT:
            continue

        center = float(np.median(residual[reliable]))
        deviation = 1.4826 * float(np.median(np.abs(residual[reliable] - center)))
        threshold = max(15.0, 0.035 * body_height, center + 3 * deviation)
        severe_threshold = max(30.0, 0.07 * body_height, center + 5 * deviation)

        outlier = (confidence[:, joint] < MINIMUM_CONFIDENCE) | (residual > threshold)
        outlier &= np.array([valid_point(point) for point in prediction[:, joint]])

        if circular_maximum_run(outlier) > 3:
            continue

        inferred_confidence = min(
            0.60,
            float(np.median(confidence[reliable, joint])) * np.exp(-center / max(threshold, 1)),
        )

        for view in np.where(outlier)[0]:
            if confidence[view, joint] >= 0.75 and residual[view] <= severe_threshold:
                continue

            if joint in {LEFT_SHOULDER, RIGHT_SHOULDER} and residual[view] <= severe_threshold:
                continue

            inlier = reliable & (residual <= threshold)

            if not reliable_neighbors(view, inlier):
                continue

            if not points_in_mask([prediction[view, joint]], masks[view], radius=5)[0]:
                continue

            refined[view, joint] = prediction[view, joint]
            refined_confidence[view, joint] = inferred_confidence
            refined_source[view, joint] = KEYPOINT_ROTATION
            refined_origin[view, joint] = -1
            replaced_views.add(view)

    return refined, refined_confidence, refined_source, refined_origin, sorted(replaced_views), prediction


def refine_arms(keypoints, confidence, source, origin_joint, masks):
    """분산된 믿을 관측이 충분한 팔에서 고립된 오류만 회전 궤적으로 채웁니다."""
    refined = keypoints.copy()
    refined_confidence = confidence.copy()
    refined_source = source.copy()
    refined_origin = origin_joint.copy()
    replaced_views = set()

    body_height = estimate_body_height(keypoints, confidence)

    for shoulder, elbow, wrist in ARM_CHAINS:
        chain = [shoulder, elbow, wrist]
        chain_confidence = confidence[:, chain].min(axis=1)
        trusted = chain_confidence >= RELIABLE_CONFIDENCE

        for view in range(VIEW):
            if not trusted[view]:
                continue

            if not points_in_mask(keypoints[view, chain], masks[view], radius=5).all():
                trusted[view] = False
                continue

            upper_length = np.linalg.norm(keypoints[view, shoulder] - keypoints[view, elbow])
            lower_length = np.linalg.norm(keypoints[view, elbow] - keypoints[view, wrist])

            if not (0.06 * body_height <= upper_length <= 0.35 * body_height):
                trusted[view] = False

            if not (0.06 * body_height <= lower_length <= 0.35 * body_height):
                trusted[view] = False

        support = np.where(trusted)[0]

        if len(support) < 6 or circular_maximum_gap(support) > ROTATION_MAXIMUM_GAP:
            continue

        fitting_confidence = np.where(trusted, chain_confidence, 0)
        prediction = np.full((VIEW, 2, 2), np.nan, dtype=np.float32)

        for index, joint in enumerate((elbow, wrist)):
            for coordinate in range(2):
                prediction[:, index, coordinate] = fit_rotation_curve(
                    keypoints[:, joint, coordinate],
                    fitting_confidence,
                    minimum_support=6,
                )

        if not np.isfinite(prediction).all():
            continue

        chain_candidates = []

        for index, joint in enumerate((elbow, wrist)):
            residual = np.linalg.norm(keypoints[:, joint] - prediction[:, index], axis=1)
            center = float(np.median(residual[trusted]))
            deviation = 1.4826 * float(np.median(np.abs(residual[trusted] - center)))
            threshold = max(20.0, 0.045 * body_height, center + 3 * deviation)
            outlier = (confidence[:, joint] < MINIMUM_CONFIDENCE) | (residual > threshold)

            for view in np.where(outlier)[0]:
                if view in SIDE_VIEWS:
                    continue

                if not reliable_neighbors(view, trusted):
                    continue

                if not points_in_mask([prediction[view, index]], masks[view], radius=5)[0]:
                    continue

                chain_candidates.append((float(residual[view]), view, joint, index))

        for _, view, joint, index in sorted(chain_candidates, reverse=True)[:2]:
            refined[view, joint] = prediction[view, index]
            refined_confidence[view, joint] = min(0.45, float(np.median(fitting_confidence[trusted])))
            refined_source[view, joint] = KEYPOINT_ARM
            refined_origin[view, joint] = -1
            replaced_views.add(view)

    return refined, refined_confidence, refined_source, refined_origin, sorted(replaced_views)


def interpolate_isolated_keypoints(keypoints, confidence, source, origin_joint, masks):
    """양옆 View가 모두 정상인 고립 결손 관절만 원형 보간합니다."""
    refined = keypoints.copy()
    refined_confidence = confidence.copy()
    refined_source = source.copy()
    refined_origin = origin_joint.copy()
    replaced_views = set()

    roi_joints = sorted({joint for values in JOINTS.values() for joint in values[:2]})
    roi_joints += [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    for joint in sorted(set(roi_joints)):
        missing = confidence[:, joint] < MINIMUM_POSE_CONFIDENCE

        if circular_maximum_run(missing) > 1:
            continue

        for view in np.where(missing)[0]:
            if joint in {LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST} and view in SIDE_VIEWS:
                continue

            previous = (view - 1) % VIEW
            following = (view + 1) % VIEW

            if min(confidence[previous, joint], confidence[following, joint]) < MINIMUM_CONFIDENCE:
                continue

            if not valid_point(keypoints[previous, joint]) or not valid_point(keypoints[following, joint]):
                continue

            point = (keypoints[previous, joint] + keypoints[following, joint]) / 2

            if not valid_point(point):
                continue

            if not points_in_mask([point], masks[view], radius=5)[0]:
                continue

            refined[view, joint] = point
            refined_confidence[view, joint] = min(
                0.40,
                float(min(confidence[previous, joint], confidence[following, joint]) * 0.5),
            )
            refined_source[view, joint] = KEYPOINT_INTERPOLATED
            refined_origin[view, joint] = -1
            replaced_views.add(view)

    return refined, refined_confidence, refined_source, refined_origin, sorted(replaced_views)


def reject_unreliable_arms(keypoints, confidence, source, origin_joint):
    """표준 TSA 자세와 맞지 않거나 측면에서 겹친 팔을 ROI 관측에서 제외합니다."""
    refined_confidence = confidence.copy()
    refined_source = source.copy()
    refined_origin = origin_joint.copy()
    rejected_views = set()

    body_height = estimate_body_height(keypoints, confidence)
    shoulder_gap = np.abs(keypoints[:, LEFT_SHOULDER, 0] - keypoints[:, RIGHT_SHOULDER, 0])
    reliable_shoulder = confidence[:, [LEFT_SHOULDER, RIGHT_SHOULDER]].min(axis=1) >= MINIMUM_CONFIDENCE
    reliable_shoulder &= np.array(
        [
            valid_point(keypoints[view, LEFT_SHOULDER]) and valid_point(keypoints[view, RIGHT_SHOULDER])
            for view in range(VIEW)
        ]
    )
    reference_gap = float(np.percentile(shoulder_gap[reliable_shoulder], 90)) if reliable_shoulder.any() else 40.0
    collapse_threshold = max(12.0, reference_gap * 0.35)

    for view in range(VIEW):
        hips = [LEFT_HIP, RIGHT_HIP]

        if confidence[view, hips].min() < MINIMUM_POSE_CONFIDENCE:
            continue

        if not all(valid_point(keypoints[view, joint]) for joint in hips):
            continue

        hip_y = float(keypoints[view, [LEFT_HIP, RIGHT_HIP], 1].mean())
        rejected_chains = set()

        for chain_index, (shoulder, elbow, wrist) in enumerate(ARM_CHAINS):
            chain = [shoulder, elbow, wrist]

            if confidence[view, chain].min() < MINIMUM_POSE_CONFIDENCE:
                rejected_chains.add(chain_index)
                continue

            if not all(valid_point(keypoints[view, joint]) for joint in chain):
                rejected_chains.add(chain_index)
                continue

            upper_length = np.linalg.norm(keypoints[view, shoulder] - keypoints[view, elbow])
            lower_length = np.linalg.norm(keypoints[view, elbow] - keypoints[view, wrist])
            raised_wrist = keypoints[view, wrist, 1] < hip_y - 0.12 * body_height
            plausible_bones = 0.05 * body_height <= upper_length <= 0.38 * body_height
            plausible_bones &= 0.05 * body_height <= lower_length <= 0.38 * body_height

            if not raised_wrist or not plausible_bones:
                rejected_chains.add(chain_index)

        elbow_gap = abs(keypoints[view, LEFT_ELBOW, 0] - keypoints[view, RIGHT_ELBOW, 0])
        wrist_gap = abs(keypoints[view, LEFT_WRIST, 0] - keypoints[view, RIGHT_WRIST, 0])

        if view in SIDE_VIEWS and min(elbow_gap, wrist_gap) < collapse_threshold:
            chain_confidence = [confidence[view, list(chain)].min() for chain in ARM_CHAINS]

            if abs(chain_confidence[0] - chain_confidence[1]) < 0.05:
                rejected_chains.update([0, 1])
            else:
                rejected_chains.add(int(np.argmin(chain_confidence)))

        for chain_index in rejected_chains:
            _, elbow, wrist = ARM_CHAINS[chain_index]
            refined_confidence[view, [elbow, wrist]] = 0
            refined_source[view, [elbow, wrist]] = KEYPOINT_INVALID
            refined_origin[view, [elbow, wrist]] = -1
            rejected_views.add(view)

    return refined_confidence, refined_source, refined_origin, sorted(rejected_views)


def pose_mask_inclusion(keypoints, confidence, masks):
    """ROI에 사용하는 관절 중 느슨한 신체 마스크 안에 위치한 비율을 계산합니다."""
    joints = sorted({joint for values in JOINTS.values() for joint in values[:2]})
    included = []

    for view in range(VIEW):
        valid = confidence[view, joints] >= MINIMUM_POSE_CONFIDENCE

        if valid.any():
            included.extend(points_in_mask(keypoints[view, np.array(joints)[valid]], masks[view], radius=5))

    return float(np.mean(included)) if included else 0.0


def pose_stage_is_safe(before_keypoints, before_confidence, keypoints, confidence, masks):
    """보정 단계가 회전·마스크·유효 관절 지표를 악화시키지 않았는지 확인합니다."""
    before_mask = pose_mask_inclusion(before_keypoints, before_confidence, masks)
    after_mask = pose_mask_inclusion(keypoints, confidence, masks)

    if after_mask < before_mask - 0.02:
        return False

    if anchor_violation_count(keypoints, confidence) > anchor_violation_count(before_keypoints, before_confidence):
        return False

    before_valid = before_confidence >= MINIMUM_POSE_CONFIDENCE
    before_valid &= np.array([[valid_point(point) for point in view] for view in before_keypoints])
    after_valid = confidence >= MINIMUM_POSE_CONFIDENCE
    after_valid &= np.array([[valid_point(point) for point in view] for view in keypoints])

    if after_valid.sum() < before_valid.sum():
        return False

    before_rotation = rotation_objective(before_keypoints, before_confidence)
    after_rotation = rotation_objective(keypoints, confidence)

    if np.isfinite(before_rotation[0]) and after_rotation[0] > before_rotation[0] * 1.05:
        return False

    if np.isfinite(before_rotation[1]) and after_rotation[1] > before_rotation[1] * 1.10:
        return False

    return True


def align_pose_labels(raw_keypoints, raw_confidence, frame_confidence):
    """좌표를 추론하지 않고 신뢰할 수 있는 좌우 label 교환만 적용합니다."""
    keypoints = raw_keypoints.copy()
    confidence = raw_confidence.copy()
    observed = np.array([[valid_point(point) for point in view] for view in keypoints])
    observed &= confidence > 0
    observed &= frame_confidence[:, None] > 0
    confidence[~observed] = 0
    source = np.where(observed, KEYPOINT_OBSERVED, KEYPOINT_INVALID).astype(np.uint8)
    origin_joint = np.broadcast_to(np.arange(KEYPOINT_COUNT), (VIEW, KEYPOINT_COUNT)).copy().astype(np.int8)
    origin_joint[~observed] = -1

    before_rotation = rotation_objective(keypoints, confidence)
    prediction = predict_rotation(keypoints, confidence)
    candidates = find_rotation_swaps(keypoints, confidence, prediction)
    candidates = [view for view in candidates if anchor_supports_swap(view, keypoints, confidence)]

    if 0 < len(candidates) <= ROTATION_MAXIMUM_SWAPS:
        swapped = apply_rotation_swaps(keypoints, confidence, source, origin_joint, candidates, prediction)
        candidate_keypoints, candidate_confidence, candidate_source, candidate_origin, _ = swapped
        after_rotation = rotation_objective(candidate_keypoints, candidate_confidence)
        anchor_before = anchor_violation_count(keypoints, confidence)
        anchor_after = anchor_violation_count(candidate_keypoints, candidate_confidence)

        if after_rotation[0] < before_rotation[0] * 0.85:
            if after_rotation[1] <= before_rotation[1] and anchor_after <= anchor_before:
                keypoints = candidate_keypoints
                confidence = candidate_confidence
                source = candidate_source
                origin_joint = candidate_origin

    aligned_keypoints, aligned_confidence, lower_views = align_lower_body(keypoints, confidence)

    for view in lower_views:
        for left, right in LOWER_BODY_JOINTS:
            if np.array_equal(aligned_keypoints[view, left], keypoints[view, left]):
                continue

            source[view, [left, right]] = KEYPOINT_SWAPPED
            origin_joint[view, [left, right]] = origin_joint[view, [right, left]]

    keypoints = aligned_keypoints
    confidence = aligned_confidence
    valid = confidence >= MINIMUM_POSE_CONFIDENCE
    valid &= np.array([[valid_point(point) for point in view] for view in keypoints])

    return keypoints, confidence, source, origin_joint, valid


def refine_pose(raw_keypoints, raw_confidence, frame_confidence, masks):
    """원본 관측을 보존하면서 좌우 오류와 명백한 회전 이상치만 정제합니다."""
    keypoints = raw_keypoints.copy()
    confidence = raw_confidence.copy()
    observed = np.array([[valid_point(point) for point in view] for view in keypoints])
    observed &= raw_confidence > 0
    observed &= frame_confidence[:, None] > 0
    confidence[~observed] = 0
    source = np.where(observed, KEYPOINT_OBSERVED, KEYPOINT_INVALID).astype(np.uint8)
    origin_joint = np.broadcast_to(np.arange(KEYPOINT_COUNT), (VIEW, KEYPOINT_COUNT)).copy().astype(np.int8)
    origin_joint[source == KEYPOINT_INVALID] = -1
    correction_flags = np.zeros(VIEW, dtype=np.uint16)

    before_rotation = rotation_objective(keypoints, confidence)
    prediction = predict_rotation(keypoints, confidence)
    candidates = find_rotation_swaps(keypoints, confidence, prediction)
    candidates = [view for view in candidates if anchor_supports_swap(view, keypoints, confidence)]

    rotation_views = []

    if 0 < len(candidates) <= ROTATION_MAXIMUM_SWAPS:
        swapped = apply_rotation_swaps(keypoints, confidence, source, origin_joint, candidates, prediction)
        candidate_keypoints, candidate_confidence, candidate_source, candidate_origin, applied = swapped
        after_rotation = rotation_objective(candidate_keypoints, candidate_confidence)
        anchor_before = anchor_violation_count(keypoints, confidence)
        anchor_after = anchor_violation_count(candidate_keypoints, candidate_confidence)

        objective_improved = after_rotation[0] < before_rotation[0] * 0.85
        tail_preserved = after_rotation[1] <= before_rotation[1]

        if objective_improved and tail_preserved and anchor_after <= anchor_before:
            keypoints = candidate_keypoints
            confidence = candidate_confidence
            source = candidate_source
            origin_joint = candidate_origin
            rotation_views = applied
            if rotation_views:
                correction_flags[rotation_views] |= ROTATION_CORRECTION

    # 전신 반전 판정이 끝난 뒤 골반을 기준으로 하체 내부의 불일치만 정리합니다.
    aligned_keypoints, aligned_confidence, lower_candidates = align_lower_body(keypoints, confidence)
    lower_source = source.copy()
    lower_origin = origin_joint.copy()

    for view in lower_candidates:
        for left, right in LOWER_BODY_JOINTS:
            if np.array_equal(aligned_keypoints[view, left], keypoints[view, left]):
                continue

            lower_source[view, [left, right]] = KEYPOINT_SWAPPED
            lower_origin[view, [left, right]] = lower_origin[view, [right, left]]

    if pose_stage_is_safe(keypoints, confidence, aligned_keypoints, aligned_confidence, masks):
        keypoints = aligned_keypoints
        confidence = aligned_confidence
        source = lower_source
        origin_joint = lower_origin
        lower_views = lower_candidates

        if lower_views:
            correction_flags[lower_views] |= LOWER_BODY_CORRECTION
    else:
        lower_views = []

    before_replacement = pose_mask_inclusion(keypoints, confidence, masks)

    replaced = replace_structural_outliers(keypoints, confidence, source, origin_joint, masks)
    candidate_keypoints, candidate_confidence, candidate_source, candidate_origin, structural_views, _ = replaced

    if pose_stage_is_safe(keypoints, confidence, candidate_keypoints, candidate_confidence, masks):
        keypoints = candidate_keypoints
        confidence = candidate_confidence
        source = candidate_source
        origin_joint = candidate_origin

        if structural_views:
            correction_flags[structural_views] |= STRUCTURAL_REPLACEMENT
    else:
        structural_views = []

    arms = refine_arms(keypoints, confidence, source, origin_joint, masks)
    candidate_keypoints, candidate_confidence, candidate_source, candidate_origin, arm_views = arms

    if pose_stage_is_safe(keypoints, confidence, candidate_keypoints, candidate_confidence, masks):
        keypoints = candidate_keypoints
        confidence = candidate_confidence
        source = candidate_source
        origin_joint = candidate_origin

        if arm_views:
            correction_flags[arm_views] |= ARM_REPLACEMENT
    else:
        arm_views = []

    interpolated = interpolate_isolated_keypoints(keypoints, confidence, source, origin_joint, masks)
    candidate_keypoints, candidate_confidence, candidate_source, candidate_origin, interpolated_views = interpolated

    if pose_stage_is_safe(keypoints, confidence, candidate_keypoints, candidate_confidence, masks):
        keypoints = candidate_keypoints
        confidence = candidate_confidence
        source = candidate_source
        origin_joint = candidate_origin

        if interpolated_views:
            correction_flags[interpolated_views] |= INTERPOLATION
    else:
        interpolated_views = []

    confidence, source, origin_joint, rejected_arm_views = reject_unreliable_arms(
        keypoints,
        confidence,
        source,
        origin_joint,
    )
    if rejected_arm_views:
        correction_flags[rejected_arm_views] |= ARM_REJECTION

    valid = confidence >= MINIMUM_POSE_CONFIDENCE
    valid &= np.array([[valid_point(point) for point in view] for view in keypoints])
    prediction = predict_rotation(keypoints, confidence)
    final_rotation = rotation_objective(keypoints, confidence)

    report = {
        "detected_views": int((frame_confidence > 0).sum()),
        "lower_body_views": lower_views,
        "rotation_candidates": candidates,
        "rotation_views": rotation_views,
        "structural_views": structural_views,
        "arm_views": arm_views,
        "rejected_arm_views": rejected_arm_views,
        "interpolated_views": interpolated_views,
        "rotation_error_before": before_rotation[0],
        "rotation_p90_before": before_rotation[1],
        "rotation_error_after": final_rotation[0],
        "rotation_p90_after": final_rotation[1],
        "mask_inclusion_before": before_replacement,
        "mask_inclusion_after": pose_mask_inclusion(keypoints, confidence, masks),
        "anchor_violations": anchor_violation_count(keypoints, confidence),
        "valid_keypoints": int(valid.sum()),
    }

    return keypoints, confidence, source, origin_joint, valid, correction_flags, prediction, report


def transform_points(points, matrix):
    """2차원 좌표에 OpenCV Affine 변환을 적용합니다."""
    points = np.asarray(points, dtype=np.float32)
    return cv2.transform(points[None], matrix)[0]


def expected_limb_width(zone, length):
    """관절 길이로 신체 구역의 기본 너비를 계산합니다."""
    ratio, minimum, maximum = LIMB_WIDTH[zone]
    return float(np.clip(length * ratio, minimum, maximum))


def polygon_mask(polygon):
    """사변형 좌표를 native 해상도의 이진 마스크로 변환합니다."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
    return mask


def clip_polygon(polygon):
    """Polygon과 이미지의 실제 교집합 면적으로 보존 비율을 계산합니다."""
    polygon = np.asarray(polygon, dtype=np.float32)
    raw_area = abs(float(cv2.contourArea(polygon)))
    frame = np.array(
        [[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]],
        dtype=np.float32,
    )
    intersection, _ = cv2.intersectConvexConvex(polygon, frame)
    clipping_ratio = float(intersection) / max(raw_area, 1e-6)

    # 네 꼭짓점은 원형을 보존하고 rasterize할 때 OpenCV가 이미지 경계에서 자릅니다.
    return polygon, float(np.clip(clipping_ratio, 0, 1))


def polygon_box(polygon):
    """사변형을 half-open Bounding Box로 변환합니다."""
    x, y, width, height = cv2.boundingRect(np.round(polygon).astype(np.int32))
    return clip_box((x, y, x + width, y + height))


def polygon_statistics(polygon, body_mask):
    """Polygon의 도형 안정성과 신체 마스크 포함 정도를 계산합니다."""
    points = np.round(polygon).astype(np.int32)
    area = abs(float(cv2.contourArea(points)))
    edges = np.roll(polygon, -1, axis=0) - polygon
    minimum_edge = float(np.linalg.norm(edges, axis=1).min())
    convex = bool(cv2.isContourConvex(points))

    mask = polygon_mask(polygon)
    foreground_ratio = float(body_mask[mask == 1].mean()) if mask.any() else 0.0

    return mask, area, minimum_edge, convex, foreground_ratio


def roi_quality(confidence, foreground_ratio, clipping_ratio, source, visibility=1.0, evidence=1.0):
    """관절, 도형, 명암 근거를 결합한 soft ROI 품질값을 계산합니다."""
    source_score = {
        ROI_INTENSITY: 1.00,
        ROI_ORIENTED: 0.75,
        ROI_TORSO: 0.85,
        ROI_RELATIVE: 0.35,
    }.get(source, 0.0)

    foreground_score = min(1.0, foreground_ratio / 0.25)
    geometry_score = min(1.0, clipping_ratio / 0.85)
    quality = 0.45 * confidence
    quality += 0.20 * foreground_score
    quality += 0.20 * geometry_score
    quality += 0.15 * source_score * evidence

    return float(np.clip(quality * visibility, 0, 1))


def limb_polygon(image, body_mask, keypoints, keypoint_confidence, zone):
    """관절과 회전된 명암 Profile을 이용하여 Limb ROI를 생성합니다."""
    start_joint, end_joint, _, _ = JOINTS[zone]
    confidence = float(keypoint_confidence[[start_joint, end_joint]].min())

    if confidence < MINIMUM_POSE_CONFIDENCE:
        return None, ROI_INVALID, {}

    start, end = segmentor(keypoints, zone)
    direction = end - start
    length = float(np.linalg.norm(direction))

    if not valid_point(start) or not valid_point(end) or length < LIMB_MINIMUM_LENGTH:
        return None, ROI_INVALID, {}

    gray = image[:, :, 0] if image.ndim == 3 else image
    gray = gray.astype(np.float32) / 255

    center = (start + end) / 2
    angle = np.degrees(np.arctan2(direction[1], direction[0])) - 90
    matrix = cv2.getRotationMatrix2D(tuple(center), float(angle), 1.0)
    inverse = cv2.invertAffineTransform(matrix)

    rotated = cv2.warpAffine(
        gray,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated = cv2.GaussianBlur(rotated, LIMB_GAUSSIAN_KERNEL, LIMB_GAUSSIAN_SIGMA)
    rotated_mask = cv2.warpAffine(
        body_mask.astype(np.uint8),
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_points = transform_points(np.stack([start, end]), matrix)
    center_x = float(rotated_points[:, 0].mean())
    margin = length * 0.05
    top = float(rotated_points[:, 1].min() - margin)
    bottom = float(rotated_points[:, 1].max() + margin)

    expected_width = expected_limb_width(zone, length)
    search_half = expected_width * LIMB_SEARCH_SCALE

    x0 = max(0, int(np.floor(center_x - search_half)))
    x1 = min(WIDTH - 1, int(np.ceil(center_x + search_half)))
    y0 = max(0, int(np.floor(top)))
    y1 = min(HEIGHT - 1, int(np.ceil(bottom)))

    use_profile = confidence >= MINIMUM_CONFIDENCE and x1 - x0 >= 7 and y1 > y0
    detected_width = expected_width
    contrast = 0.0
    center_offset = 0.0
    profile_range = 0.0
    profile_snr = 0.0

    if use_profile:
        raw_profile = rotated[y0 : y1 + 1, x0 : x1 + 1].mean(axis=0)
        lower, upper = np.percentile(raw_profile, [5, 95])
        profile_range = float(upper - lower)
        difference = np.diff(raw_profile)
        noise = 1.4826 * float(np.median(np.abs(difference - np.median(difference))))
        profile_snr = profile_range / max(noise, 1e-6)
        use_profile = profile_range >= 0.02 and profile_snr >= 3.0
        profile = np.clip((raw_profile - lower) / max(profile_range, 1e-6), 0, 1)
        mask_profile = rotated_mask[y0 : y1 + 1, x0 : x1 + 1].mean(axis=0)
        horizontal = np.linspace(-1, 1, len(profile))
        parameter = np.polyfit(horizontal, profile, LIMB_POLYNOMIAL_DEGREE)
        fitted = np.polyval(parameter, horizontal)

        minima = np.where((fitted[1:-1] <= fitted[:-2]) & (fitted[1:-1] < fitted[2:]))[0] + 1
        local_center = center_x - x0
        minimum_half = expected_width * 0.25
        left_candidates = minima[minima < local_center - minimum_half]
        right_candidates = minima[minima > local_center + minimum_half]

        # 느슨한 신체 마스크가 중심을 가로지르는 경우 명암 경계의 과도한 확장을 제한합니다.
        supported = mask_profile >= 0.15
        transitions = np.diff(np.pad(supported.astype(np.int8), (1, 1)))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        mask_candidate = None

        if len(starts):
            distances = [
                0 if start <= local_center < end else min(abs(local_center - start), abs(local_center - end))
                for start, end in zip(starts, ends)
            ]
            selected = int(np.argmin(distances))
            mask_left = float(x0 + starts[selected])
            mask_right = float(x0 + ends[selected])
            mask_width = mask_right - mask_left
            mask_offset = abs((mask_left + mask_right) / 2 - center_x)

            inside = profile[starts[selected] : ends[selected]]
            outside = np.concatenate([profile[: starts[selected]], profile[ends[selected] :]])
            mask_contrast = float(inside.mean() - np.median(outside)) if len(outside) else 0.0

            if expected_width * 0.45 <= mask_width <= expected_width * 1.80:
                if mask_offset <= expected_width * 0.35 and mask_contrast >= LIMB_MINIMUM_CONTRAST:
                    mask_candidate = (mask_left, mask_right, mask_contrast, mask_offset)

        if mask_candidate is not None:
            left, right, contrast, center_offset = mask_candidate
            detected_width = right - left
        elif len(left_candidates) == 0 or len(right_candidates) == 0:
            use_profile = False
        else:
            left_index = int(left_candidates[-1])
            right_index = int(right_candidates[0])
            left = float(x0 + left_index)
            right = float(x0 + right_index)
            detected_width = right - left
            boundary = float((fitted[left_index] + fitted[right_index]) / 2)
            contrast = float(fitted[left_index : right_index + 1].mean() - boundary)
            center_offset = abs((left + right) / 2 - center_x)

            if detected_width < expected_width * 0.45:
                use_profile = False
            elif detected_width > expected_width * 1.80:
                use_profile = False
            elif center_offset > expected_width * 0.35:
                use_profile = False
            elif contrast < LIMB_MINIMUM_CONTRAST:
                use_profile = False

    if use_profile:
        source = ROI_INTENSITY
    else:
        left = center_x - expected_width / 2
        right = center_x + expected_width / 2
        detected_width = expected_width
        contrast = 0.0
        center_offset = 0.0
        source = ROI_ORIENTED

    rectangle = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )
    polygon = transform_points(rectangle, inverse)
    polygon, clipping_ratio = clip_polygon(polygon)
    mask, area, minimum_edge, convex, foreground_ratio = polygon_statistics(polygon, body_mask)

    if clipping_ratio < 0.70 or area < 50 or minimum_edge < 4 or not convex:
        return None, ROI_INVALID, {}

    endpoint_containment = all(
        cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point.astype(float)), False) >= 0
        for point in (start, end)
    )

    if not endpoint_containment:
        return None, ROI_INVALID, {}

    if source == ROI_INTENSITY:
        evidence = min(1.0, profile_range / 0.10)
        evidence *= min(1.0, profile_snr / 8.0)
        evidence *= min(1.0, contrast / 0.15)
    else:
        evidence = 0.60

    quality = roi_quality(confidence, foreground_ratio, clipping_ratio, source, evidence=evidence)

    metrics = {
        "confidence": confidence,
        "length": length,
        "width": detected_width,
        "width_ratio": detected_width / max(length, 1e-6),
        "center_offset": center_offset,
        "contrast": contrast,
        "profile_range": profile_range,
        "profile_snr": profile_snr,
        "foreground_ratio": foreground_ratio,
        "clipping_ratio": clipping_ratio,
        "endpoint_containment": endpoint_containment,
        "quality": quality,
        "visible": confidence >= MINIMUM_CONFIDENCE,
        "visibility": 1.0 if confidence >= MINIMUM_CONFIDENCE else 0.35,
        "reliable": confidence >= MINIMUM_CONFIDENCE and source != ROI_INVALID,
        "mask": mask,
    }

    return polygon, source, metrics


def midpoint(first, second):
    """두 관절의 중점을 계산합니다."""
    return (first + second) / 2


def scale_width(left, right, scale):
    """두 점의 중심은 유지하면서 구역의 가로 너비를 조절합니다."""
    center = midpoint(left, right)
    return center + (left - center) * scale, center + (right - center) * scale


def torso_strip(left_shoulder, right_shoulder, left_hip, right_hip, top, bottom, width_scale):
    """어깨-골반 기둥에서 상하 Edge를 잘라 Torso 사변형을 만듭니다."""
    left_top = compensator(left_shoulder, left_hip, top)
    right_top = compensator(right_shoulder, right_hip, top)
    left_bottom = compensator(left_shoulder, left_hip, bottom)
    right_bottom = compensator(right_shoulder, right_hip, bottom)

    left_top, right_top = scale_width(left_top, right_top, width_scale)
    left_bottom, right_bottom = scale_width(left_bottom, right_bottom, width_scale)

    return np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.float32)


def torso_visibility(zone, view):
    """정면 Chest와 후면 Upper Back을 서로 다른 View에서 활성화합니다."""
    if zone not in {5, 17}:
        return 1.0

    if view in TORSO_PRIMARY_VIEWS[zone]:
        return 1.0

    if view in TORSO_SUPPORT_VIEWS[zone]:
        return 0.35

    return 0.0


def torso_polygon_points(keypoints, zone):
    """Zone별 Pillar와 Edge Shift로 Torso Polygon 좌표를 계산합니다."""
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]

    neck = midpoint(left_shoulder, right_shoulder)
    pelvis = midpoint(left_hip, right_hip)

    if zone == 5:
        return torso_strip(left_shoulder, right_shoulder, left_hip, right_hip, 0.08, 0.56, 0.92)

    if zone == 17:
        return torso_strip(left_shoulder, right_shoulder, left_hip, right_hip, 0.00, 0.62, 1.04)

    if zone == 6:
        inner_top = compensator(neck, pelvis, 0.35)
        outer_top = compensator(right_shoulder, right_hip, 0.35)
        return np.array([inner_top, outer_top, right_hip, pelvis], dtype=np.float32)

    if zone == 7:
        outer_top = compensator(left_shoulder, left_hip, 0.35)
        inner_top = compensator(neck, pelvis, 0.35)
        return np.array([outer_top, inner_top, pelvis, left_hip], dtype=np.float32)

    left_knee = keypoints[LEFT_KNEE]
    right_knee = keypoints[RIGHT_KNEE]
    left_thigh = compensator(left_hip, left_knee, 0.22)
    right_thigh = compensator(right_hip, right_knee, 0.22)
    lower_center = midpoint(left_thigh, right_thigh)

    left_top = pelvis + (left_hip - pelvis) * 0.70
    right_top = pelvis + (right_hip - pelvis) * 0.70
    left_bottom = lower_center + (left_thigh - lower_center) * 0.55
    right_bottom = lower_center + (right_thigh - lower_center) * 0.55

    return np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.float32)


def torso_polygon(body_mask, keypoints, keypoint_confidence, zone, view):
    """Pose Estimator의 네 기둥 좌표로 Torso ROI를 생성합니다."""
    visibility = torso_visibility(zone, view)

    if visibility == 0:
        return None, ROI_INVALID, {"visibility": 0.0}

    joints = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    if zone == 9:
        joints += [LEFT_KNEE, RIGHT_KNEE]

    confidence = float(keypoint_confidence[joints].min())

    if confidence < MINIMUM_POSE_CONFIDENCE:
        return None, ROI_INVALID, {"visibility": visibility}

    polygon = torso_polygon_points(keypoints, zone)

    if confidence < MINIMUM_CONFIDENCE:
        center = polygon.mean(axis=0)
        polygon = center + (polygon - center) * 1.08

    polygon, clipping_ratio = clip_polygon(polygon)
    mask, area, minimum_edge, convex, foreground_ratio = polygon_statistics(polygon, body_mask)

    if clipping_ratio < 0.70 or area < TORSO_MINIMUM_AREA or minimum_edge < TORSO_MINIMUM_EDGE or not convex:
        return None, ROI_INVALID, {"visibility": visibility}

    quality = roi_quality(confidence, foreground_ratio, clipping_ratio, ROI_TORSO, visibility)
    metrics = {
        "confidence": confidence,
        "foreground_ratio": foreground_ratio,
        "clipping_ratio": clipping_ratio,
        "contrast": 0.0,
        "quality": quality,
        "visible": visibility == 1.0,
        "visibility": visibility,
        "reliable": confidence >= MINIMUM_CONFIDENCE,
        "mask": mask,
    }

    return polygon, ROI_TORSO, metrics


def relative_polygon(body_box, zone, view):
    """Pose가 없을 때 신체 Bounding Box 안에서 보수적인 기본 ROI를 생성합니다."""
    if np.any(np.asarray(body_box) < 0):
        return None

    if zone in {5, 17} and torso_visibility(zone, view) == 0:
        return None

    x1, y1, x2, y2 = body_box.astype(float)
    body_width = x2 - x1
    body_height = y2 - y1
    top, bottom, width_ratio, side = ZONE_ANCHORS[zone]

    center_x = (x1 + x2) / 2
    direction = {"left": -1, "right": 1, "center": 0}[side]
    center_x += direction * (-np.cos(VIEW_ANGLES[view])) * body_width * 0.25
    roi_width = body_width * width_ratio

    polygon = np.array(
        [
            [center_x - roi_width / 2, y1 + body_height * top],
            [center_x + roi_width / 2, y1 + body_height * top],
            [center_x + roi_width / 2, y1 + body_height * bottom],
            [center_x - roi_width / 2, y1 + body_height * bottom],
        ],
        dtype=np.float32,
    )

    polygon, clipping_ratio = clip_polygon(polygon)

    if clipping_ratio < 0.70:
        return None

    return polygon, clipping_ratio


def build_rois(images, masks, keypoints, keypoint_confidence, body_boxes):
    """16개 View의 17개 신체 구역 Polygon과 품질 정보를 생성합니다."""
    polygons = np.full((VIEW, ZONE_COUNT, 4, 2), -1, dtype=np.float32)
    boxes = np.full((VIEW, ZONE_COUNT, 4), -1, dtype=np.int16)
    valid = np.zeros((VIEW, ZONE_COUNT), dtype=bool)
    visible = np.zeros((VIEW, ZONE_COUNT), dtype=bool)
    visibility = np.zeros((VIEW, ZONE_COUNT), dtype=np.float32)
    reliable = np.zeros((VIEW, ZONE_COUNT), dtype=bool)
    quality = np.zeros((VIEW, ZONE_COUNT), dtype=np.float32)
    source = np.zeros((VIEW, ZONE_COUNT), dtype=np.uint8)
    mask_support = np.zeros((VIEW, ZONE_COUNT), dtype=np.float32)
    profile_contrast = np.zeros((VIEW, ZONE_COUNT), dtype=np.float32)
    profile_range = np.zeros((VIEW, ZONE_COUNT), dtype=np.float32)
    profile_snr = np.zeros((VIEW, ZONE_COUNT), dtype=np.float32)

    for view in range(VIEW):
        for zone in range(1, ZONE_COUNT + 1):
            if zone in JOINTS:
                polygon, method, metrics = limb_polygon(
                    images[view],
                    masks[view],
                    keypoints[view],
                    keypoint_confidence[view],
                    zone,
                )
            else:
                polygon, method, metrics = torso_polygon(
                    masks[view],
                    keypoints[view],
                    keypoint_confidence[view],
                    zone,
                    view,
                )

            if polygon is None and not (zone in {5, 17} and torso_visibility(zone, view) == 0):
                fallback = relative_polygon(body_boxes[view], zone, view)

                if fallback is not None:
                    polygon, clipping_ratio = fallback
                    method = ROI_RELATIVE
                    roi_mask, _, _, _, foreground_ratio = polygon_statistics(polygon, masks[view])
                    if zone in {5, 17}:
                        view_support = torso_visibility(zone, view)
                    elif view in SIDE_VIEWS:
                        view_support = 0.15
                    else:
                        view_support = 0.50

                    metrics = {
                        "confidence": 0.0,
                        "foreground_ratio": foreground_ratio,
                        "clipping_ratio": clipping_ratio,
                        "contrast": 0.0,
                        "profile_range": 0.0,
                        "profile_snr": 0.0,
                        "quality": roi_quality(
                            0.0,
                            foreground_ratio,
                            clipping_ratio,
                            ROI_RELATIVE,
                            view_support,
                        ),
                        "visible": view_support >= 0.50,
                        "visibility": view_support,
                        "reliable": False,
                        "mask": roi_mask,
                    }

            if polygon is None:
                continue

            index = zone - 1
            polygons[view, index] = polygon
            boxes[view, index] = polygon_box(polygon)
            valid[view, index] = True
            visible[view, index] = bool(metrics.get("visible", False))
            visibility[view, index] = float(metrics.get("visibility", 0))
            reliable[view, index] = bool(metrics.get("reliable", False))
            quality[view, index] = float(metrics.get("quality", 0))
            source[view, index] = method
            mask_support[view, index] = float(metrics.get("foreground_ratio", 0))
            profile_contrast[view, index] = float(metrics.get("contrast", 0))
            profile_range[view, index] = float(metrics.get("profile_range", 0))
            profile_snr[view, index] = float(metrics.get("profile_snr", 0))

    return {
        "polygons": polygons,
        "boxes": boxes,
        "roi_valid": valid,
        "roi_visible": visible,
        "roi_visibility": visibility,
        "roi_reliable": reliable,
        "roi_quality": quality,
        "roi_source": source,
        "mask_support": mask_support,
        "profile_contrast": profile_contrast,
        "profile_range": profile_range,
        "profile_snr": profile_snr,
    }


def visualize(images, keypoints, keypoint_confidence, frame_confidence, outfile, threshold=0.5):
    """16개 이미지 View 위에 Pose Estimator가 검출한 관절을 시각화합니다."""
    figure, axes = plt.subplots(4, 4, figsize=(12, 16))

    for view, axis in enumerate(axes.flat):
        axis.imshow(images[view])

        # 신뢰도가 높은 관절끼리 연결하여 오탐된 관절로 인한 오류를 방지합니다.
        for start, end in SKELETON:
            if keypoint_confidence[view, start] < threshold:
                continue

            if keypoint_confidence[view, end] < threshold:
                continue

            x = [keypoints[view, start, 0], keypoints[view, end, 0]]
            y = [keypoints[view, start, 1], keypoints[view, end, 1]]

            axis.plot(x, y, color="yellow", linewidth=1)

        confidence = keypoint_confidence[view]
        reliable_keypoint = confidence >= threshold
        uncertain_keypoint = (confidence > 0) & (confidence < threshold)

        axis.scatter(
            keypoints[view, reliable_keypoint, 0],
            keypoints[view, reliable_keypoint, 1],
            color="cyan",
            s=10,
        )

        # 검출에 성공하였지만 신뢰도가 낮아 이후 조정이 필요해 보이는 관절입니다.
        axis.scatter(
            keypoints[view, uncertain_keypoint, 0],
            keypoints[view, uncertain_keypoint, 1],
            color="red",
            s=10,
        )

        axis.set_title(f"View {view} | Confidence {frame_confidence[view]:.2f}")
        axis.axis("off")

    output = Path(outfile)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure.suptitle("Yellow: Reliable Keypoint | Red: Uncertain Keypoint")
    figure.tight_layout()
    figure.savefig(output, dpi=150)

    plt.close(figure)


def visualize_rotation(keypoints, keypoint_confidence, prediction, outfile):
    """관절의 x좌표와 회전 곡선으로 예측 계산한 좌표와 비교합니다."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    views = np.arange(VIEW)

    for axis, (name, (left, right)) in zip(axes.flat, ROTATION_JOINTS.items()):
        left_size = 20 + keypoint_confidence[:, left] * 40
        right_size = 20 + keypoint_confidence[:, right] * 40

        # YOLO의 관측 좌표를 표시합니다. 점이 클수록 신뢰도가 높은 점입니다.
        axis.scatter(views, keypoints[:, left, 0], color="royalblue", s=left_size)
        axis.scatter(views, keypoints[:, right, 0], color="darkorange", s=right_size)

        # 예측 계산한 회전 궤도를 선으로 시각화합니다.
        axis.plot(views, prediction[:, left, 0], color="royalblue", label="Left")
        axis.plot(views, prediction[:, right, 0], color="darkorange", label="Right")

        axis.set_title(name)
        axis.set_ylabel("X Coordinate")
        axis.set_xticks(views)
        axis.grid(alpha=0.3)
        axis.legend()

    output = Path(outfile)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output, dpi=150)

    plt.close(figure)


def visualize_rois(images, roi_data, outfile):
    """16개 View에 생성된 17개 ROI Polygon을 시각화합니다."""
    figure, axes = plt.subplots(4, 4, figsize=(12, 16))
    colors = plt.cm.tab20(np.linspace(0, 1, ZONE_COUNT))

    for view, axis in enumerate(axes.flat):
        axis.imshow(images[view])

        for zone in range(ZONE_COUNT):
            if not roi_data["roi_valid"][view, zone]:
                continue

            polygon = roi_data["polygons"][view, zone]
            closed = np.vstack([polygon, polygon[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=colors[zone], linewidth=1)
            center = polygon.mean(axis=0)
            axis.text(center[0], center[1], str(zone + 1), color="white", fontsize=6)

        valid = int(roi_data["roi_valid"][view].sum())
        pose = int(np.isin(roi_data["roi_source"][view], [ROI_INTENSITY, ROI_ORIENTED, ROI_TORSO]).sum())
        axis.set_title(f"View {view} | Valid {valid} | Pose {pose}")
        axis.axis("off")

    output = Path(outfile)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def file_sha256(infile):
    """파일의 SHA256 값을 계산합니다."""
    digest = hashlib.sha256()

    with open(infile, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(outfile, data):
    """JSON 파일을 임시 경로에 기록한 뒤 원자적으로 교체합니다."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_name(f".{outfile.name}.{os.getpid()}.tmp")

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temporary, outfile)


def write_csv(outfile, data):
    """CSV 파일을 임시 경로에 기록한 뒤 원자적으로 교체합니다."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_name(f".{outfile.name}.{os.getpid()}.tmp")
    data.to_csv(temporary, index=False)
    os.replace(temporary, outfile)


def write_text(outfile, data):
    """Text 파일을 임시 경로에 기록한 뒤 원자적으로 교체합니다."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_name(f".{outfile.name}.{os.getpid()}.tmp")

    with open(temporary, "w", encoding="utf-8") as file:
        file.write(data)

    os.replace(temporary, outfile)


def pipeline_signature(model_hash, source_hash=None):
    """실제 ROI 생성 설정으로 재현 가능한 Pipeline 식별자를 만듭니다."""
    source_hash = source_hash or file_sha256(Path(__file__))
    configuration = {
        "artifact_version": ARTIFACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "model_sha256": model_hash,
        "source_sha256": source_hash,
        "image_shape": [HEIGHT, WIDTH, VIEW],
        "clahe": [CLAHE_LIMIT, *CLAHE_GRID],
        "body_mask_percentile": BODY_MASK_PERCENTILE,
        "confidence": [MINIMUM_POSE_CONFIDENCE, MINIMUM_CONFIDENCE, RELIABLE_CONFIDENCE],
        "rotation": [
            HUBER_DELTA,
            ROTATION_FIT_ITERATIONS,
            ROTATION_PAIR_SWAP_RATIO,
            ROTATION_TOTAL_SWAP_RATIO,
            ROTATION_MINIMUM_SUPPORT,
            ROTATION_MAXIMUM_GAP,
            ROTATION_MAXIMUM_SWAPS,
            ROTATION_MINIMUM_VOTES,
        ],
        "structural_joints": STRUCTURAL_JOINTS,
        "arm_chains": ARM_CHAINS,
        "side_views": sorted(SIDE_VIEWS),
        "joint_zones": JOINTS,
        "correction_flags": [
            LOWER_BODY_CORRECTION,
            ROTATION_CORRECTION,
            STRUCTURAL_REPLACEMENT,
            ARM_REPLACEMENT,
            INTERPOLATION,
            ARM_REJECTION,
        ],
        "limb_width": LIMB_WIDTH,
        "limb_profile": [
            *LIMB_GAUSSIAN_KERNEL,
            LIMB_GAUSSIAN_SIGMA,
            LIMB_POLYNOMIAL_DEGREE,
            LIMB_SEARCH_SCALE,
            LIMB_MINIMUM_CONTRAST,
            LIMB_MINIMUM_LENGTH,
        ],
        "torso_primary_views": {zone: sorted(views) for zone, views in TORSO_PRIMARY_VIEWS.items()},
        "torso_support_views": {zone: sorted(views) for zone, views in TORSO_SUPPORT_VIEWS.items()},
        "zone_anchors": ZONE_ANCHORS,
        "torso_shape_thresholds": [TORSO_MINIMUM_AREA, TORSO_MINIMUM_EDGE],
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def serializable_report(report):
    """NumPy 수치와 무한값을 JSON에 안전한 값으로 변환합니다."""

    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()

        if isinstance(value, float) and not np.isfinite(value):
            return None

        if isinstance(value, np.ndarray):
            return convert(value.tolist())

        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]

        return value

    return convert(report)


def process_scan(model, scan_id, aps_path, signature="", model_hash="", return_context=False):
    """한 피험자의 Pose와 17개 Zone ROI artifact를 생성합니다."""
    images, masks = adapter(aps_path)
    raw_keypoints, raw_confidence, frame_confidence, body_boxes = detector(model, images)

    refined = refine_pose(raw_keypoints, raw_confidence, frame_confidence, masks)
    keypoints, confidence, keypoint_source, origin_joint, keypoint_valid, correction_flags, prediction, report = refined
    roi_data = build_rois(images, masks, keypoints, confidence, body_boxes)

    report["valid_rois"] = int(roi_data["roi_valid"].sum())
    report["pose_rois"] = int(np.isin(roi_data["roi_source"], [ROI_INTENSITY, ROI_ORIENTED, ROI_TORSO]).sum())
    report["relative_rois"] = int((roi_data["roi_source"] == ROI_RELATIVE).sum())
    valid_quality = roi_data["roi_quality"][roi_data["roi_valid"]]
    report["mean_roi_quality"] = float(valid_quality.mean()) if len(valid_quality) else 0.0
    source_hash = file_sha256(Path(__file__))

    artifact = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int16),
        "artifact_version": np.array(ARTIFACT_VERSION),
        "pipeline_signature": np.array(signature),
        "model_sha256": np.array(model_hash),
        "source_sha256": np.array(source_hash),
        "scan_id": np.array(str(scan_id)),
        "raw_keypoints": raw_keypoints.astype(np.float32),
        "raw_keypoint_confidence": raw_confidence.astype(np.float32),
        "keypoints": keypoints.astype(np.float32),
        "keypoint_confidence": confidence.astype(np.float32),
        "keypoint_source": keypoint_source.astype(np.uint8),
        "origin_joint": origin_joint.astype(np.int8),
        "keypoint_valid": keypoint_valid.astype(bool),
        "frame_confidence": frame_confidence.astype(np.float32),
        "body_boxes": body_boxes.astype(np.int16),
        "correction_flags": correction_flags.astype(np.uint16),
        "rotation_prediction": prediction.astype(np.float32),
        "pose_report": np.array(json.dumps(serializable_report(report), ensure_ascii=False)),
        **roi_data,
    }

    if return_context:
        return artifact, images, masks, report

    return artifact


CHECKPOINT_SHAPES = {
    "raw_keypoints": (VIEW, KEYPOINT_COUNT, 2),
    "raw_keypoint_confidence": (VIEW, KEYPOINT_COUNT),
    "keypoints": (VIEW, KEYPOINT_COUNT, 2),
    "keypoint_confidence": (VIEW, KEYPOINT_COUNT),
    "keypoint_source": (VIEW, KEYPOINT_COUNT),
    "origin_joint": (VIEW, KEYPOINT_COUNT),
    "keypoint_valid": (VIEW, KEYPOINT_COUNT),
    "frame_confidence": (VIEW,),
    "body_boxes": (VIEW, 4),
    "correction_flags": (VIEW,),
    "rotation_prediction": (VIEW, KEYPOINT_COUNT, 2),
    "polygons": (VIEW, ZONE_COUNT, 4, 2),
    "boxes": (VIEW, ZONE_COUNT, 4),
    "roi_valid": (VIEW, ZONE_COUNT),
    "roi_visible": (VIEW, ZONE_COUNT),
    "roi_visibility": (VIEW, ZONE_COUNT),
    "roi_reliable": (VIEW, ZONE_COUNT),
    "roi_quality": (VIEW, ZONE_COUNT),
    "roi_source": (VIEW, ZONE_COUNT),
    "mask_support": (VIEW, ZONE_COUNT),
    "profile_contrast": (VIEW, ZONE_COUNT),
    "profile_range": (VIEW, ZONE_COUNT),
    "profile_snr": (VIEW, ZONE_COUNT),
}

CHECKPOINT_DTYPES = {
    "raw_keypoints": np.float32,
    "raw_keypoint_confidence": np.float32,
    "keypoints": np.float32,
    "keypoint_confidence": np.float32,
    "keypoint_source": np.uint8,
    "origin_joint": np.int8,
    "keypoint_valid": np.bool_,
    "frame_confidence": np.float32,
    "body_boxes": np.int16,
    "correction_flags": np.uint16,
    "rotation_prediction": np.float32,
    "polygons": np.float32,
    "boxes": np.int16,
    "roi_valid": np.bool_,
    "roi_visible": np.bool_,
    "roi_visibility": np.float32,
    "roi_reliable": np.bool_,
    "roi_quality": np.float32,
    "roi_source": np.uint8,
    "mask_support": np.float32,
    "profile_contrast": np.float32,
    "profile_range": np.float32,
    "profile_snr": np.float32,
}

FORBIDDEN_ARTIFACT_KEYS = {"labels", "split", "type", "Probability"}
CHECKPOINT_METADATA_KEYS = {
    "schema_version",
    "artifact_version",
    "pipeline_signature",
    "model_sha256",
    "source_sha256",
    "scan_id",
    "pose_report",
}


def valid_sha256(value):
    """문자열이 64자리 SHA256 형식인지 확인합니다."""
    value = str(value)
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_keypoint_semantics(data):
    """관절의 좌표, 신뢰도, 출처, 원본 관절 메타데이터를 함께 검사합니다."""
    errors = []
    keypoints = np.asarray(data["keypoints"])
    confidence = np.asarray(data["keypoint_confidence"])
    source = np.asarray(data["keypoint_source"])
    origin = np.asarray(data["origin_joint"])
    valid = np.asarray(data["keypoint_valid"])

    inside = np.isfinite(keypoints).all(axis=-1)
    inside &= (keypoints[..., 0] >= 0) & (keypoints[..., 0] < WIDTH)
    inside &= (keypoints[..., 1] >= 0) & (keypoints[..., 1] < HEIGHT)
    expected_valid = inside & (confidence >= MINIMUM_POSE_CONFIDENCE)

    if not np.array_equal(valid, expected_valid):
        errors.append("keypoint validity does not match confidence and coordinates")

    joint_indices = np.arange(KEYPOINT_COUNT, dtype=np.int8)
    joint_indices = np.broadcast_to(joint_indices, source.shape)
    observed = source == KEYPOINT_OBSERVED
    swapped = source == KEYPOINT_SWAPPED
    inferred = np.isin(source, [KEYPOINT_ROTATION, KEYPOINT_ARM, KEYPOINT_INTERPOLATED])
    invalid = source == KEYPOINT_INVALID

    if np.any(observed & ((confidence <= 0) | (origin != joint_indices))):
        errors.append("observed keypoint metadata is inconsistent")

    if np.any(swapped & ((confidence <= 0) | (origin < 0) | (origin >= KEYPOINT_COUNT))):
        errors.append("swapped keypoint metadata is inconsistent")

    if np.any(inferred & ((confidence <= 0) | (origin != -1))):
        errors.append("inferred keypoint metadata is inconsistent")

    if np.any(invalid & ((confidence != 0) | valid | (origin != -1))):
        errors.append("invalid keypoint metadata is inconsistent")

    prediction = np.asarray(data["rotation_prediction"])

    if prediction.ndim == 3:
        prediction = prediction[None]

    for scan in range(len(prediction)):
        for joint in range(KEYPOINT_COUNT):
            values = prediction[scan, :, joint]

            if not (np.isfinite(values).all() or np.isnan(values).all()):
                errors.append("rotation prediction must be entirely finite or NaN for each joint")
                return errors

    return errors


def validate_roi_semantics(data):
    """ROI polygon, half-open box, 가시성, 출처 메타데이터를 함께 검사합니다."""
    errors = []
    polygons = np.asarray(data["polygons"])
    boxes = np.asarray(data["boxes"])
    valid = np.asarray(data["roi_valid"])
    visible = np.asarray(data["roi_visible"])
    visibility = np.asarray(data["roi_visibility"])
    reliable = np.asarray(data["roi_reliable"])
    source = np.asarray(data["roi_source"])

    if not np.array_equal(visible, valid & (visibility >= 0.50)):
        errors.append("ROI visible flag does not match visibility score")

    if np.any(reliable & np.isin(source, [ROI_INVALID, ROI_RELATIVE])):
        errors.append("relative or invalid ROI cannot be reliable")

    valid_polygons = polygons[valid]
    valid_boxes = boxes[valid]
    valid_sources = source[valid]

    for polygon, box, method in zip(valid_polygons, valid_boxes, valid_sources):
        points = np.round(polygon).astype(np.int32)
        area = abs(float(cv2.contourArea(points)))
        edges = np.roll(polygon, -1, axis=0) - polygon
        minimum_edge = float(np.linalg.norm(edges, axis=1).min())
        _, clipping_ratio = clip_polygon(polygon)

        if area < 50 or minimum_edge < 4 or not cv2.isContourConvex(points):
            errors.append("valid ROI polygon has invalid geometry")
            break

        if method == ROI_TORSO and (area < TORSO_MINIMUM_AREA or minimum_edge < TORSO_MINIMUM_EDGE):
            errors.append("torso ROI polygon has invalid geometry")
            break

        if clipping_ratio < 0.70:
            errors.append("valid ROI polygon has insufficient image intersection")
            break

        if not np.array_equal(box, polygon_box(polygon)):
            errors.append("ROI box does not match polygon")
            break

    if np.any(np.asarray(data["profile_contrast"]) < 0):
        errors.append("profile_contrast below zero")

    return errors


def validate_pose_report(report, data):
    """Pose 처리 보고서가 실제 배열과 correction flag를 반영하는지 검사합니다."""
    errors = []
    required = {
        "detected_views",
        "lower_body_views",
        "rotation_candidates",
        "rotation_views",
        "structural_views",
        "arm_views",
        "rejected_arm_views",
        "interpolated_views",
        "rotation_error_before",
        "rotation_p90_before",
        "rotation_error_after",
        "rotation_p90_after",
        "mask_inclusion_before",
        "mask_inclusion_after",
        "anchor_violations",
        "valid_keypoints",
        "valid_rois",
        "pose_rois",
        "relative_rois",
        "mean_roi_quality",
    }

    missing = required - set(report)

    if missing:
        return [f"pose report missing keys: {sorted(missing)}"]

    list_flags = {
        "lower_body_views": LOWER_BODY_CORRECTION,
        "rotation_views": ROTATION_CORRECTION,
        "structural_views": STRUCTURAL_REPLACEMENT,
        "arm_views": ARM_REPLACEMENT,
        "rejected_arm_views": ARM_REJECTION,
        "interpolated_views": INTERPOLATION,
    }
    correction_flags = np.asarray(data["correction_flags"])

    for key in ["rotation_candidates", *list_flags]:
        views = report[key]

        if not isinstance(views, list) or len(views) != len(set(views)):
            errors.append(f"pose report {key} must contain unique views")
            continue

        if any(not isinstance(view, int) or not 0 <= view < VIEW for view in views):
            errors.append(f"pose report {key} contains invalid view")

    for key, flag in list_flags.items():
        expected = set(np.where((correction_flags & flag) > 0)[0].tolist())

        if set(report[key]) != expected:
            errors.append(f"pose report {key} disagrees with correction flags")

    expected_values = {
        "detected_views": int((np.asarray(data["frame_confidence"]) > 0).sum()),
        "valid_keypoints": int(np.asarray(data["keypoint_valid"]).sum()),
        "valid_rois": int(np.asarray(data["roi_valid"]).sum()),
        "pose_rois": int(np.isin(np.asarray(data["roi_source"]), [ROI_INTENSITY, ROI_ORIENTED, ROI_TORSO]).sum()),
        "relative_rois": int((np.asarray(data["roi_source"]) == ROI_RELATIVE).sum()),
    }

    for key, value in expected_values.items():
        if report[key] != value:
            errors.append(f"pose report {key} disagrees with artifact")

    return errors


def validate_checkpoint(data, expected_scan_id=None, expected_signature=None):
    """피험자별 Checkpoint의 shape, dtype, 좌표 계약을 검사합니다."""
    errors = []
    keys = set(data.files if hasattr(data, "files") else data)

    scalar_keys = sorted(CHECKPOINT_METADATA_KEYS)

    for key in scalar_keys:
        if key not in keys:
            errors.append(f"missing key: {key}")

    for key in set(scalar_keys) & keys:
        if data[key].shape != ():
            errors.append(f"{key} must be a scalar")

        if data[key].dtype == object:
            errors.append(f"{key} uses object dtype")

    for key, shape in CHECKPOINT_SHAPES.items():
        if key not in keys:
            errors.append(f"missing key: {key}")
            continue

        if data[key].shape != shape:
            errors.append(f"{key} shape: {data[key].shape} != {shape}")

        if data[key].dtype != np.dtype(CHECKPOINT_DTYPES[key]):
            errors.append(f"{key} dtype: {data[key].dtype} != {np.dtype(CHECKPOINT_DTYPES[key])}")

    unknown_keys = keys - CHECKPOINT_METADATA_KEYS - set(CHECKPOINT_SHAPES)

    if unknown_keys:
        errors.append(f"unknown artifact keys: {sorted(unknown_keys)}")

    if errors:
        return errors

    if data["schema_version"].dtype != np.dtype(np.int16) or int(data["schema_version"]) != SCHEMA_VERSION:
        errors.append("schema version mismatch")

    if str(data["artifact_version"]) != ARTIFACT_VERSION:
        errors.append("artifact version mismatch")

    if expected_scan_id is not None and str(data["scan_id"]) != str(expected_scan_id):
        errors.append("scan id mismatch")

    if expected_signature is not None and str(data["pipeline_signature"]) != expected_signature:
        errors.append("pipeline signature mismatch")

    if not valid_sha256(data["pipeline_signature"]):
        errors.append("invalid pipeline signature")

    if not valid_sha256(data["model_sha256"]):
        errors.append("invalid model hash")

    if not valid_sha256(data["source_sha256"]):
        errors.append("invalid source hash")
    elif str(data["pipeline_signature"]) != pipeline_signature(
        str(data["model_sha256"]),
        str(data["source_sha256"]),
    ):
        errors.append("pipeline signature does not match model and source hashes")

    if not str(data["scan_id"]):
        errors.append("empty scan id")

    try:
        pose_report = json.loads(str(data["pose_report"]))

        if not isinstance(pose_report, dict):
            errors.append("pose report must be a JSON object")
        else:
            errors.extend(validate_pose_report(pose_report, data))
    except json.JSONDecodeError:
        errors.append("pose report is not valid JSON")

    if keys & FORBIDDEN_ARTIFACT_KEYS or any(key.startswith("zone_") for key in keys):
        errors.append("classification label leaked into artifact")

    finite_keys = [
        "raw_keypoints",
        "raw_keypoint_confidence",
        "keypoints",
        "keypoint_confidence",
        "frame_confidence",
        "roi_quality",
        "roi_visibility",
        "mask_support",
        "profile_contrast",
        "profile_range",
        "profile_snr",
    ]

    for key in finite_keys:
        if not np.isfinite(data[key]).all():
            errors.append(f"{key} contains NaN or Inf")

    for key in [
        "raw_keypoint_confidence",
        "keypoint_confidence",
        "frame_confidence",
        "roi_quality",
        "roi_visibility",
    ]:
        if np.any((data[key] < 0) | (data[key] > 1)):
            errors.append(f"{key} outside [0, 1]")

    for key in ["mask_support", "profile_range"]:
        if np.any((data[key] < 0) | (data[key] > 1)):
            errors.append(f"{key} outside [0, 1]")

    if np.any(data["profile_snr"] < 0):
        errors.append("profile_snr below zero")

    valid_keypoints = data["keypoints"][data["keypoint_valid"]]

    if valid_keypoints.size:
        if np.any((valid_keypoints[:, 0] < 0) | (valid_keypoints[:, 0] >= WIDTH)):
            errors.append("valid keypoint x outside image")

        if np.any((valid_keypoints[:, 1] < 0) | (valid_keypoints[:, 1] >= HEIGHT)):
            errors.append("valid keypoint y outside image")

    valid_boxes = data["boxes"][data["roi_valid"]]
    invalid_boxes = data["boxes"][~data["roi_valid"]]
    valid_polygons = data["polygons"][data["roi_valid"]]
    invalid_polygons = data["polygons"][~data["roi_valid"]]

    if valid_boxes.size:
        x1, y1, x2, y2 = valid_boxes.T

        if np.any((x1 < 0) | (y1 < 0) | (x2 > WIDTH) | (y2 > HEIGHT) | (x1 >= x2) | (y1 >= y2)):
            errors.append("valid ROI box outside half-open image bounds")

    if invalid_boxes.size and not np.all(invalid_boxes == -1):
        errors.append("invalid ROI box does not use -1 sentinel")

    if valid_polygons.size and not np.isfinite(valid_polygons).all():
        errors.append("valid ROI polygon contains NaN or Inf")

    if invalid_polygons.size and not np.all(invalid_polygons == -1):
        errors.append("invalid ROI polygon does not use -1 sentinel")

    if np.any(data["roi_visible"] & ~data["roi_valid"]):
        errors.append("visible ROI is not valid")

    if np.any(data["roi_reliable"] & ~data["roi_valid"]):
        errors.append("reliable ROI is not valid")

    invalid_rois = ~data["roi_valid"]

    for key in ["roi_visibility", "roi_quality", "mask_support", "profile_contrast", "profile_range", "profile_snr"]:
        if np.any(data[key][invalid_rois] != 0):
            errors.append(f"invalid ROI has nonzero {key}")

    if np.any(data["keypoint_valid"] & (data["keypoint_source"] == KEYPOINT_INVALID)):
        errors.append("valid keypoint has invalid source")

    invalid_keypoints = data["keypoint_source"] == KEYPOINT_INVALID
    inferred_keypoints = np.isin(
        data["keypoint_source"],
        [KEYPOINT_ROTATION, KEYPOINT_ARM, KEYPOINT_INTERPOLATED],
    )

    if np.any(data["keypoint_confidence"][invalid_keypoints] != 0):
        errors.append("invalid keypoint has nonzero confidence")

    if np.any(data["origin_joint"][invalid_keypoints | inferred_keypoints] != -1):
        errors.append("invalid or inferred keypoint has observed origin")

    if np.any((data["origin_joint"] < -1) | (data["origin_joint"] >= KEYPOINT_COUNT)):
        errors.append("origin_joint outside -1..16")

    if np.any((data["roi_source"] == ROI_INVALID) != ~data["roi_valid"]):
        errors.append("ROI source and validity disagree")

    valid_body_boxes = data["body_boxes"][np.all(data["body_boxes"] >= 0, axis=1)]
    invalid_body_boxes = data["body_boxes"][np.any(data["body_boxes"] < 0, axis=1)]

    if valid_body_boxes.size:
        x1, y1, x2, y2 = valid_body_boxes.T

        if np.any((x1 < 0) | (y1 < 0) | (x2 > WIDTH) | (y2 > HEIGHT) | (x1 >= x2) | (y1 >= y2)):
            errors.append("body box outside half-open image bounds")

    if invalid_body_boxes.size and not np.all(invalid_body_boxes == -1):
        errors.append("invalid body box does not use -1 sentinel")

    if np.any(data["keypoint_source"] > KEYPOINT_INTERPOLATED):
        errors.append("unknown keypoint source")

    if np.any(data["roi_source"] > ROI_RELATIVE):
        errors.append("unknown ROI source")

    known_flags = (
        LOWER_BODY_CORRECTION
        | ROTATION_CORRECTION
        | STRUCTURAL_REPLACEMENT
        | ARM_REPLACEMENT
        | INTERPOLATION
        | ARM_REJECTION
    )

    unknown_flag_mask = np.uint16(~known_flags & 0xFFFF)

    if np.any(data["correction_flags"] & unknown_flag_mask):
        errors.append("unknown correction flag")

    errors.extend(validate_keypoint_semantics(data))
    errors.extend(validate_roi_semantics(data))

    return errors


def save_checkpoint(outfile, artifact):
    """Checkpoint를 임시 파일에 저장한 뒤 원자적으로 교체합니다."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_name(f".{outfile.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **artifact)

    with np.load(temporary, allow_pickle=False) as saved:
        errors = validate_checkpoint(saved, str(artifact["scan_id"]), str(artifact["pipeline_signature"]))

    if errors:
        temporary.unlink(missing_ok=True)
        raise ValueError("Invalid checkpoint: " + "; ".join(errors))

    os.replace(temporary, outfile)


def dataset_records(dataset_file):
    """Dataset에서 label을 제외한 scan ID와 APS 경로만 정렬해 불러옵니다."""
    dataset = pd.read_csv(dataset_file, usecols=["scan_id", "aps_path"])

    if dataset["scan_id"].duplicated().any():
        raise ValueError("dataset.csv contains duplicate scan_id")

    return dataset.sort_values("scan_id").reset_index(drop=True)


def export_artifacts(
    dataset_file,
    data_directory,
    model_file,
    output_directory,
    shard_index=0,
    shard_count=1,
    limit=None,
    force=False,
):
    """누락된 피험자만 처리하여 개별 Checkpoint로 저장합니다."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be inside 0..shard_count-1")

    if limit is not None and limit < 0:
        raise ValueError("limit must be zero or greater")

    records = dataset_records(dataset_file).iloc[shard_index::shard_count]

    if limit is not None:
        records = records.iloc[:limit]

    model_file = Path(model_file)
    model_hash = file_sha256(model_file)
    signature = pipeline_signature(model_hash)
    checkpoint_directory = Path(output_directory) / "checkpoints"
    rows = list(records.itertuples(index=False))
    pending = []

    for current, row in enumerate(rows, start=1):
        checkpoint = checkpoint_directory / f"{row.scan_id}.npz"

        if checkpoint.exists() and not force:
            with np.load(checkpoint, allow_pickle=False) as saved:
                errors = validate_checkpoint(saved, row.scan_id, signature)

            if errors:
                raise ValueError(f"{checkpoint}: " + "; ".join(errors))

            print(f"[{current}/{len(records)}] {row.scan_id}: Skipped")
            continue

        pending.append((current, row, checkpoint))

    if not pending:
        return

    model = YOLO(model_file)

    for current, row, checkpoint in pending:
        aps_path = Path(data_directory) / row.aps_path
        artifact = process_scan(model, row.scan_id, aps_path, signature, model_hash)
        save_checkpoint(checkpoint, artifact)
        report = json.loads(str(artifact["pose_report"]))
        print(
            f"[{current}/{len(records)}] {row.scan_id}: "
            f"Views {report['detected_views']}/{VIEW} | ROI {report['valid_rois']}/{VIEW * ZONE_COUNT}"
        )


def checkpoints_match(first, second):
    """같은 scan ID의 두 Checkpoint 내용이 완전히 같은지 확인합니다."""
    with np.load(first, allow_pickle=False) as first_data, np.load(second, allow_pickle=False) as second_data:
        if set(first_data.files) != set(second_data.files):
            return False

        for key in first_data.files:
            first_array = first_data[key]
            second_array = second_data[key]
            equal_nan = first_array.dtype.kind in {"f", "c"}

            if not np.array_equal(first_array, second_array, equal_nan=equal_nan):
                return False

        return True


def collect_checkpoints(dataset_file, source_directories, output_directory):
    """여러 Workspace의 shard Checkpoint를 한 디렉터리로 안전하게 모읍니다."""
    allowed_scan_ids = set(dataset_records(dataset_file)["scan_id"].astype(str))
    destination = Path(output_directory) / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    signature = None
    copied = 0
    skipped = 0

    for source_directory in source_directories:
        source_directory = Path(source_directory)

        if (source_directory / "checkpoints").is_dir():
            source_directory /= "checkpoints"

        checkpoints = sorted(path for path in source_directory.glob("*.npz") if not path.name.startswith("."))

        if not checkpoints:
            raise ValueError(f"No checkpoints found in {source_directory}")

        for checkpoint in checkpoints:
            scan_id = checkpoint.stem

            if scan_id not in allowed_scan_ids:
                raise ValueError(f"Unknown scan ID: {scan_id}")

            with np.load(checkpoint, allow_pickle=False) as saved:
                current_signature = str(saved["pipeline_signature"]) if "pipeline_signature" in saved else ""
                errors = validate_checkpoint(saved, scan_id, signature or current_signature)

            if errors:
                raise ValueError(f"{checkpoint}: " + "; ".join(errors))

            signature = current_signature
            outfile = destination / checkpoint.name

            if outfile.exists():
                if not checkpoints_match(checkpoint, outfile):
                    raise ValueError(f"Conflicting checkpoint: {scan_id}")

                skipped += 1
                continue

            temporary = outfile.with_name(f".{outfile.stem}.{os.getpid()}.tmp.npz")
            shutil.copy2(checkpoint, temporary)
            os.replace(temporary, outfile)
            copied += 1

    print(f"Collected: {copied} | Identical skipped: {skipped} | Total: {len(list(destination.glob('*.npz')))}")


AGGREGATE_KEYS = [
    "raw_keypoints",
    "raw_keypoint_confidence",
    "keypoints",
    "keypoint_confidence",
    "keypoint_source",
    "origin_joint",
    "keypoint_valid",
    "frame_confidence",
    "body_boxes",
    "correction_flags",
    "rotation_prediction",
    "polygons",
    "boxes",
    "roi_valid",
    "roi_visible",
    "roi_visibility",
    "roi_reliable",
    "roi_quality",
    "roi_source",
    "mask_support",
    "profile_contrast",
    "profile_range",
    "profile_snr",
]


def validate_aggregate(data, expected_scan_ids=None, allowed_scan_ids=None):
    """통합 NPZ artifact의 순서와 배열 계약을 검사합니다."""
    errors = []
    keys = set(data.files if hasattr(data, "files") else data)

    scalar_keys = ["schema_version", "artifact_version", "pipeline_signature", "model_sha256", "source_sha256"]
    allowed_keys = set(scalar_keys) | {"scan_ids"} | set(AGGREGATE_KEYS)

    for key in [*scalar_keys, "scan_ids"]:
        if key not in keys:
            errors.append(f"missing key: {key}")

    for key in AGGREGATE_KEYS:
        if key not in keys:
            errors.append(f"missing key: {key}")

    for key in set(scalar_keys) & keys:
        if data[key].shape != ():
            errors.append(f"{key} must be a scalar")

        if data[key].dtype == object:
            errors.append(f"{key} uses object dtype")

    if "scan_ids" in keys and data["scan_ids"].ndim != 1:
        errors.append("scan_ids must be one-dimensional")

    if "scan_ids" in keys and data["scan_ids"].dtype == object:
        errors.append("scan_ids uses object dtype")

    unknown_keys = keys - allowed_keys

    if unknown_keys:
        errors.append(f"unknown aggregate keys: {sorted(unknown_keys)}")

    if errors:
        return errors

    scan_ids = data["scan_ids"].astype(str)
    count = len(scan_ids)

    if count == 0:
        errors.append("aggregate contains no scans")

    if len(set(scan_ids)) != count:
        errors.append("duplicate scan_id in aggregate")

    if list(scan_ids) != sorted(scan_ids):
        errors.append("scan_ids are not sorted")

    if expected_scan_ids is not None and list(scan_ids) != sorted(map(str, expected_scan_ids)):
        errors.append("aggregate scan_ids do not match dataset")

    if allowed_scan_ids is not None and not set(scan_ids).issubset(set(map(str, allowed_scan_ids))):
        errors.append("aggregate contains scan_id outside dataset")

    shape_errors = []

    for key in AGGREGATE_KEYS:
        expected_shape = (count, *CHECKPOINT_SHAPES[key])

        if data[key].shape != expected_shape:
            shape_errors.append(f"{key} shape: {data[key].shape} != {expected_shape}")

        if data[key].dtype != np.dtype(CHECKPOINT_DTYPES[key]):
            shape_errors.append(f"{key} dtype mismatch")

    errors.extend(shape_errors)

    if shape_errors:
        return errors

    if keys & FORBIDDEN_ARTIFACT_KEYS or any(key.startswith("zone_") for key in keys):
        errors.append("classification label leaked into aggregate")

    if data["schema_version"].dtype != np.dtype(np.int16) or int(data["schema_version"]) != SCHEMA_VERSION:
        errors.append("schema version mismatch")

    if str(data["artifact_version"]) != ARTIFACT_VERSION:
        errors.append("artifact version mismatch")

    if not valid_sha256(data["pipeline_signature"]):
        errors.append("invalid pipeline signature")

    if not valid_sha256(data["model_sha256"]):
        errors.append("invalid model hash")

    if not valid_sha256(data["source_sha256"]):
        errors.append("invalid source hash")
    elif str(data["pipeline_signature"]) != pipeline_signature(
        str(data["model_sha256"]),
        str(data["source_sha256"]),
    ):
        errors.append("pipeline signature does not match model and source hashes")

    for key in ["mask_support", "profile_range"]:
        if np.any((data[key] < 0) | (data[key] > 1)):
            errors.append(f"{key} outside [0, 1]")

    if np.any(data["profile_snr"] < 0):
        errors.append("profile_snr below zero")

    valid_keypoints = data["keypoints"][data["keypoint_valid"]]

    if valid_keypoints.size:
        if np.any((valid_keypoints[:, 0] < 0) | (valid_keypoints[:, 0] >= WIDTH)):
            errors.append("valid keypoint x outside image")

        if np.any((valid_keypoints[:, 1] < 0) | (valid_keypoints[:, 1] >= HEIGHT)):
            errors.append("valid keypoint y outside image")

    valid_boxes = data["boxes"][data["roi_valid"]]
    invalid_boxes = data["boxes"][~data["roi_valid"]]
    valid_polygons = data["polygons"][data["roi_valid"]]
    invalid_polygons = data["polygons"][~data["roi_valid"]]

    if valid_boxes.size:
        x1, y1, x2, y2 = valid_boxes.T

        if np.any((x1 < 0) | (y1 < 0) | (x2 > WIDTH) | (y2 > HEIGHT) | (x1 >= x2) | (y1 >= y2)):
            errors.append("valid ROI box outside half-open image bounds")

    if invalid_boxes.size and not np.all(invalid_boxes == -1):
        errors.append("invalid ROI box does not use -1 sentinel")

    if valid_polygons.size and not np.isfinite(valid_polygons).all():
        errors.append("valid ROI polygon contains NaN or Inf")

    if invalid_polygons.size and not np.all(invalid_polygons == -1):
        errors.append("invalid ROI polygon does not use -1 sentinel")

    finite_keys = [
        "raw_keypoints",
        "raw_keypoint_confidence",
        "keypoints",
        "keypoint_confidence",
        "frame_confidence",
        "roi_quality",
        "roi_visibility",
        "mask_support",
        "profile_contrast",
        "profile_range",
        "profile_snr",
    ]

    for key in finite_keys:
        if not np.isfinite(data[key]).all():
            errors.append(f"{key} contains NaN or Inf")

    for key in [
        "raw_keypoint_confidence",
        "keypoint_confidence",
        "frame_confidence",
        "roi_quality",
        "roi_visibility",
    ]:
        if np.any((data[key] < 0) | (data[key] > 1)):
            errors.append(f"{key} outside [0, 1]")

    if np.any(data["roi_visible"] & ~data["roi_valid"]):
        errors.append("visible ROI is not valid")

    if np.any(data["roi_reliable"] & ~data["roi_valid"]):
        errors.append("reliable ROI is not valid")

    invalid_rois = ~data["roi_valid"]

    for key in ["roi_visibility", "roi_quality", "mask_support", "profile_contrast", "profile_range", "profile_snr"]:
        if np.any(data[key][invalid_rois] != 0):
            errors.append(f"invalid ROI has nonzero {key}")

    if np.any(data["keypoint_valid"] & (data["keypoint_source"] == KEYPOINT_INVALID)):
        errors.append("valid keypoint has invalid source")

    if np.any((data["origin_joint"] < -1) | (data["origin_joint"] >= KEYPOINT_COUNT)):
        errors.append("origin_joint outside -1..16")

    if np.any((data["roi_source"] == ROI_INVALID) != ~data["roi_valid"]):
        errors.append("ROI source and validity disagree")

    invalid_keypoints = data["keypoint_source"] == KEYPOINT_INVALID
    inferred_keypoints = np.isin(
        data["keypoint_source"],
        [KEYPOINT_ROTATION, KEYPOINT_ARM, KEYPOINT_INTERPOLATED],
    )

    if np.any(data["keypoint_confidence"][invalid_keypoints] != 0):
        errors.append("invalid keypoint has nonzero confidence")

    if np.any(data["origin_joint"][invalid_keypoints | inferred_keypoints] != -1):
        errors.append("invalid or inferred keypoint has observed origin")

    if np.any(data["keypoint_source"] > KEYPOINT_INTERPOLATED):
        errors.append("unknown keypoint source")

    if np.any(data["roi_source"] > ROI_RELATIVE):
        errors.append("unknown ROI source")

    known_flags = (
        LOWER_BODY_CORRECTION
        | ROTATION_CORRECTION
        | STRUCTURAL_REPLACEMENT
        | ARM_REPLACEMENT
        | INTERPOLATION
        | ARM_REJECTION
    )
    unknown_flag_mask = np.uint16(~known_flags & 0xFFFF)

    if np.any(data["correction_flags"] & unknown_flag_mask):
        errors.append("unknown correction flag")

    valid_body_boxes = data["body_boxes"][np.all(data["body_boxes"] >= 0, axis=2)]
    invalid_body_boxes = data["body_boxes"][np.any(data["body_boxes"] < 0, axis=2)]

    if valid_body_boxes.size:
        x1, y1, x2, y2 = valid_body_boxes.T

        if np.any((x1 < 0) | (y1 < 0) | (x2 > WIDTH) | (y2 > HEIGHT) | (x1 >= x2) | (y1 >= y2)):
            errors.append("body box outside half-open image bounds")

    if invalid_body_boxes.size and not np.all(invalid_body_boxes == -1):
        errors.append("invalid body box does not use -1 sentinel")

    errors.extend(validate_keypoint_semantics(data))
    errors.extend(validate_roi_semantics(data))

    return errors


def finalize_artifacts(dataset_file, output_directory, allow_partial=False):
    """전체 Checkpoint를 검증하고 scan ID 순서의 단일 NPZ로 병합합니다."""
    records = dataset_records(dataset_file)
    expected_ids = records["scan_id"].astype(str).tolist()
    checkpoint_directory = Path(output_directory) / "checkpoints"
    available = {path.stem: path for path in checkpoint_directory.glob("*.npz") if not path.name.startswith(".")}
    missing = sorted(set(expected_ids) - set(available))
    extra = sorted(set(available) - set(expected_ids))

    if extra:
        raise ValueError(f"Unknown checkpoints: {extra[:5]}")

    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} checkpoints. First: {missing[:5]}")

    selected_ids = sorted(set(expected_ids) & set(available))

    if not selected_ids:
        raise ValueError("No checkpoints to finalize")

    arrays = {key: [] for key in AGGREGATE_KEYS}
    signature = None
    model_hash = None
    source_hash = None

    for scan_id in selected_ids:
        with np.load(available[scan_id], allow_pickle=False) as saved:
            current_signature = str(saved["pipeline_signature"])
            errors = validate_checkpoint(saved, scan_id, current_signature)

            if errors:
                raise ValueError(f"{available[scan_id]}: " + "; ".join(errors))

            current_model_hash = str(saved["model_sha256"])

            if model_hash is not None and current_model_hash != model_hash:
                raise ValueError("Checkpoints use different pose model weights")

            model_hash = current_model_hash
            current_source_hash = str(saved["source_sha256"])

            if source_hash is not None and current_source_hash != source_hash:
                raise ValueError("Checkpoints use different ROI source code")

            if signature is not None and current_signature != signature:
                raise ValueError("Checkpoints use different ROI pipeline settings")

            signature = current_signature
            source_hash = current_source_hash

            for key in AGGREGATE_KEYS:
                arrays[key].append(saved[key].copy())

    aggregate = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int16),
        "artifact_version": np.array(ARTIFACT_VERSION),
        "pipeline_signature": np.array(signature),
        "model_sha256": np.array(model_hash),
        "source_sha256": np.array(source_hash),
        "scan_ids": np.asarray(selected_ids),
        **{key: np.stack(values) for key, values in arrays.items()},
    }

    suffix = ".partial" if missing else ""
    outfile = Path(output_directory) / f"{ARTIFACT_VERSION}{suffix}.npz"
    temporary = outfile.with_name(f".{outfile.stem}.{os.getpid()}.tmp.npz")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **aggregate)

    with np.load(temporary, allow_pickle=False) as saved:
        expected = selected_ids if allow_partial else expected_ids
        errors = validate_aggregate(saved, expected)

    if errors:
        temporary.unlink(missing_ok=True)
        raise ValueError("Invalid aggregate: " + "; ".join(errors))

    os.replace(temporary, outfile)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": {
            "origin": "top_left",
            "height": HEIGHT,
            "width": WIDTH,
            "bbox": "half_open",
            "polygon": "four raw corners before image clipping",
            "classifier_padding": "append one row below y=659 only after loading",
        },
        "view_order": "original_aps_0_to_15",
        "view_angle_degrees": 360 / VIEW,
        "scan_count": len(selected_ids),
        "partial": bool(missing),
        "model_sha256": model_hash,
        "source_sha256": source_hash,
        "pipeline_signature": signature,
        "artifact_sha256": file_sha256(outfile),
        "keypoint_source_codes": {
            "0": "invalid",
            "1": "observed",
            "2": "swapped_observation",
            "3": "rotation_replacement",
            "4": "arm_replacement",
            "5": "circular_interpolation",
        },
        "roi_source_codes": {
            "0": "invalid_or_not_visible",
            "1": "paper_intensity_profile",
            "2": "oriented_pose_fallback",
            "3": "paper_inspired_torso",
            "4": "body_relative_fallback",
        },
        "roi_fields": {
            "roi_valid": "a usable polygon and box were generated",
            "roi_visible": "the zone is a primary or strongly supported view",
            "roi_visibility": "soft view support in the range 0 to 1",
            "roi_reliable": "localized from pose rather than body-relative fallback",
            "roi_quality": "self-consistency score, not an accuracy estimate",
        },
    }

    manifest_name = "manifest.partial.json" if missing else "manifest.json"
    manifest_file = Path(output_directory) / manifest_name

    write_json(manifest_file, manifest)

    print(f"Finalized: {outfile}")
    print(f"Scans: {len(selected_ids)} | Missing: {len(missing)}")
    return outfile


def audit_artifacts(dataset_file, artifact_file, output_directory, allow_partial=False):
    """학습 전 구조 계약과 GT 없는 자기일관성 품질 지표를 검사합니다."""
    records = dataset_records(dataset_file)
    dataset_ids = records["scan_id"].astype(str).tolist()
    expected_ids = None if allow_partial else dataset_ids

    artifact_file = Path(artifact_file)

    with np.load(artifact_file, allow_pickle=False) as data:
        errors = validate_aggregate(data, expected_ids, dataset_ids)
        reports = []

        if not errors:
            scan_ids = data["scan_ids"].astype(str)

        for index, scan_id in enumerate(scan_ids if not errors else []):
            detected_views = int((data["frame_confidence"][index] > 0).sum())
            correction_flags = data["correction_flags"][index]
            roi_valid = data["roi_valid"][index]
            roi_source = data["roi_source"][index]
            valid_quality = data["roi_quality"][index][roi_valid]
            valid_polygons = data["polygons"][index][roi_valid]

            boundary_touch = 0

            if len(valid_polygons):
                boundary_touch = int(
                    np.any(
                        (valid_polygons[:, :, 0] <= 0)
                        | (valid_polygons[:, :, 0] >= WIDTH - 1)
                        | (valid_polygons[:, :, 1] <= 0)
                        | (valid_polygons[:, :, 1] >= HEIGHT - 1),
                        axis=1,
                    ).sum()
                )

            warnings = []

            if detected_views < 12:
                warnings.append("detected_views_below_12")

            if int(roi_valid.sum()) < 230:
                warnings.append("valid_rois_below_230")

            if int((roi_source == ROI_RELATIVE).sum()) > 40:
                warnings.append("relative_fallback_above_40")

            if int(((correction_flags & ROTATION_CORRECTION) > 0).sum()) > 3:
                warnings.append("rotation_swaps_above_3")

            if int(((correction_flags & ARM_REJECTION) > 0).sum()) > 6:
                warnings.append("arm_rejections_above_6")

            if boundary_touch > 30:
                warnings.append("boundary_touch_above_30")

            reports.append(
                {
                    "scan_id": scan_id,
                    "detected_views": detected_views,
                    "mean_frame_confidence": float(data["frame_confidence"][index].mean()),
                    "lower_body_views": int(((correction_flags & LOWER_BODY_CORRECTION) > 0).sum()),
                    "rotation_views": int(((correction_flags & ROTATION_CORRECTION) > 0).sum()),
                    "structural_views": int(((correction_flags & STRUCTURAL_REPLACEMENT) > 0).sum()),
                    "arm_views": int(((correction_flags & ARM_REPLACEMENT) > 0).sum()),
                    "rejected_arm_views": int(((correction_flags & ARM_REJECTION) > 0).sum()),
                    "interpolated_views": int(((correction_flags & INTERPOLATION) > 0).sum()),
                    "valid_keypoints": int(data["keypoint_valid"][index].sum()),
                    "valid_rois": int(roi_valid.sum()),
                    "intensity_rois": int((roi_source == ROI_INTENSITY).sum()),
                    "oriented_rois": int((roi_source == ROI_ORIENTED).sum()),
                    "torso_rois": int((roi_source == ROI_TORSO).sum()),
                    "relative_rois": int((roi_source == ROI_RELATIVE).sum()),
                    "mean_roi_quality": float(valid_quality.mean()) if len(valid_quality) else 0.0,
                    "low_mask_support": int((data["mask_support"][index][roi_valid] < 0.10).sum()),
                    "boundary_touch": boundary_touch,
                    "warning_count": len(warnings),
                    "warnings": "|".join(warnings),
                }
            )

        if not errors:
            manifest_name = "manifest.partial.json" if allow_partial else "manifest.json"
            manifest_file = artifact_file.parent / manifest_name

            if not manifest_file.exists():
                errors.append(f"missing manifest: {manifest_name}")
            else:
                try:
                    with open(manifest_file, encoding="utf-8") as file:
                        manifest = json.load(file)
                except (json.JSONDecodeError, OSError) as error:
                    errors.append(f"invalid manifest: {error}")
                else:
                    manifest_checks = {
                        "artifact_version": str(data["artifact_version"]),
                        "schema_version": int(data["schema_version"]),
                        "pipeline_signature": str(data["pipeline_signature"]),
                        "model_sha256": str(data["model_sha256"]),
                        "source_sha256": str(data["source_sha256"]),
                        "scan_count": len(scan_ids),
                        "partial": bool(allow_partial),
                        "artifact_sha256": file_sha256(artifact_file),
                    }

                    for key, value in manifest_checks.items():
                        if manifest.get(key) != value:
                            errors.append(f"manifest {key} mismatch")

    report_frame = pd.DataFrame(reports)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / "audit.csv", report_frame)

    scan_count = len(report_frame)
    detected_view_rate = 0.0
    valid_roi_rate = 0.0
    mean_roi_quality = 0.0
    subjects_with_warning = 0

    if scan_count:
        detected_view_rate = float(report_frame["detected_views"].sum() / (scan_count * VIEW))
        valid_roi_rate = float(report_frame["valid_rois"].sum() / (scan_count * VIEW * ZONE_COUNT))
        mean_roi_quality = float(report_frame["mean_roi_quality"].mean())
        subjects_with_warning = int((report_frame["warning_count"] > 0).sum())

    audit = {
        "all_checks_passed": not errors,
        "hard_errors": errors,
        "self_consistency_is_not_accuracy": True,
        "summary": {
            "scan_count": scan_count,
            "detected_view_rate": detected_view_rate,
            "valid_roi_rate": valid_roi_rate,
            "mean_roi_quality": mean_roi_quality,
            "subjects_with_warning": subjects_with_warning,
        },
    }

    write_json(output_directory / "audit.json", audit)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def preview_samples(dataset_file, data_directory, model_file, output_directory, sample_count=10):
    """고정된 Training 표본에서 Pose 정제와 ROI 결과를 시각화합니다."""
    dataset = pd.read_csv(dataset_file)
    train_data = dataset[dataset["type"] == "train"]
    samples = train_data.sample(n=min(sample_count, len(train_data)), random_state=42)

    model_file = Path(model_file)
    model_hash = file_sha256(model_file)
    signature = pipeline_signature(model_hash)
    model = YOLO(model_file)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    reports = []

    for row in samples.itertuples(index=False):
        aps_path = Path(data_directory) / row.aps_path
        artifact, images, _, report = process_scan(
            model,
            row.scan_id,
            aps_path,
            signature,
            model_hash,
            return_context=True,
        )

        visualize(
            images,
            artifact["raw_keypoints"],
            artifact["raw_keypoint_confidence"],
            artifact["frame_confidence"],
            output_directory / "raw" / f"{row.scan_id}.png",
        )
        visualize(
            images,
            artifact["keypoints"],
            artifact["keypoint_confidence"],
            artifact["frame_confidence"],
            output_directory / "refined" / f"{row.scan_id}.png",
        )
        visualize_rotation(
            artifact["keypoints"],
            artifact["keypoint_confidence"],
            artifact["rotation_prediction"],
            output_directory / "rotation" / f"{row.scan_id}.png",
        )
        visualize_rois(images, artifact, output_directory / "roi" / f"{row.scan_id}.png")

        reports.append({"scan_id": row.scan_id, **serializable_report(report)})
        print(
            f"{row.scan_id}: Views {report['detected_views']}/{VIEW} | "
            f"ROI {report['valid_rois']}/{VIEW * ZONE_COUNT} | Rotation {report['rotation_views']}"
        )

    write_json(output_directory / "preview_report.json", reports)


def balanced_sample(data, sample_count, random_seed, group_column="difficulty_quartile"):
    """Pose 난이도 사분위에서 같은 수에 가깝게 scan을 추출합니다."""
    if sample_count < 1 or sample_count > len(data):
        raise ValueError("sample_count must be inside the available scan range")

    generator = np.random.default_rng(random_seed)
    groups = sorted(data[group_column].unique())
    base, remainder = divmod(sample_count, len(groups))
    selected = []

    for position, group in enumerate(groups):
        candidates = data[data[group_column] == group]
        count = base + int(position < remainder)

        if count > len(candidates):
            raise ValueError(f"Not enough scans in {group_column} {group}")

        indices = generator.choice(candidates.index.to_numpy(), count, replace=False)
        selected.append(data.loc[indices])

    return pd.concat(selected).sort_values([group_column, "scan_id"]).reset_index(drop=True)


def ground_truth_candidates(dataset_file, artifact_file):
    """분류 label을 사용하지 않고 Raw Pose 난이도 사분위를 계산합니다."""
    dataset = pd.read_csv(dataset_file, usecols=["scan_id", "aps_path", "type"])
    dataset["scan_id"] = dataset["scan_id"].astype(str)
    training = dataset[dataset["type"] == "train"].copy()
    training["scan_id"] = training["scan_id"].astype(str)
    training_scan_ids = set(training["scan_id"])

    with np.load(artifact_file, allow_pickle=False) as data:
        errors = validate_aggregate(data, allowed_scan_ids=dataset["scan_id"])

        if errors:
            raise ValueError("Invalid artifact: " + "; ".join(errors))

        rows = []

        for index, scan_id in enumerate(data["scan_ids"].astype(str)):
            if scan_id not in training_scan_ids:
                continue

            confidence = data["raw_keypoint_confidence"][index, :, POSE_JOINTS]
            frame_confidence = data["frame_confidence"][index]
            body_boxes = data["body_boxes"][index]
            valid_boxes = body_boxes[np.all(body_boxes >= 0, axis=1)]
            body_height = np.median(valid_boxes[:, 3] - valid_boxes[:, 1]) if len(valid_boxes) else 0
            low_view_rate = float((np.median(confidence, axis=1) < MINIMUM_CONFIDENCE).mean())
            detection_rate = float((frame_confidence > 0).mean())
            median_confidence = float(np.median(confidence))
            scale_penalty = 1 - float(np.clip(body_height / 500, 0, 1))
            roi_source = data["roi_source"][index]
            roi_valid = data["roi_valid"][index]
            valid_quality = data["roi_quality"][index][roi_valid]
            relative_rate = float((roi_source == ROI_RELATIVE).sum() / max(1, roi_valid.sum()))
            intensity_rate = float((roi_source == ROI_INTENSITY).sum() / max(1, roi_valid.sum()))
            mean_roi_quality = float(valid_quality.mean()) if len(valid_quality) else 0.0
            difficulty = 0.55 * (1 - median_confidence)
            difficulty += 0.30 * max(low_view_rate, 1 - detection_rate)
            difficulty += 0.15 * scale_penalty
            roi_difficulty = 0.60 * relative_rate + 0.40 * (1 - mean_roi_quality)

            rows.append(
                {
                    "scan_id": scan_id,
                    "median_pose_confidence": median_confidence,
                    "low_confidence_view_rate": low_view_rate,
                    "detected_view_rate": detection_rate,
                    "median_body_height": float(body_height),
                    "difficulty_score": difficulty,
                    "relative_roi_rate": relative_rate,
                    "intensity_roi_rate": intensity_rate,
                    "mean_roi_quality": mean_roi_quality,
                    "roi_difficulty_score": roi_difficulty,
                }
            )

    difficulty = pd.DataFrame(rows)

    if len(difficulty) < 4:
        raise ValueError("At least four training scans are required for difficulty sampling")

    ranks = difficulty["difficulty_score"].rank(method="first")
    difficulty["difficulty_quartile"] = pd.qcut(ranks, 4, labels=False).astype(int)
    roi_ranks = difficulty["roi_difficulty_score"].rank(method="first")
    difficulty["roi_difficulty_quartile"] = pd.qcut(roi_ranks, 4, labels=False).astype(int)

    return training.merge(difficulty, on="scan_id", how="inner")


def assign_evaluation_set(samples, random_seed):
    """각 난이도 사분위 안에서 calibration과 locked test를 나눕니다."""
    generator = np.random.default_rng(random_seed)
    samples = samples.copy()
    samples["evaluation_set"] = "calibration"

    for group in sorted(samples["difficulty_quartile"].unique()):
        indices = samples.index[samples["difficulty_quartile"] == group].to_numpy(copy=True)
        generator.shuffle(indices)
        locked_count = max(1, int(round(len(indices) * 0.40))) if len(indices) > 1 else 0
        samples.loc[indices[:locked_count], "evaluation_set"] = "locked"

    target_locked = int(round(len(samples) * 0.40))
    current_locked = int((samples["evaluation_set"] == "locked").sum())

    if current_locked < target_locked:
        candidates = samples.index[samples["evaluation_set"] == "calibration"].to_numpy(copy=True)
        generator.shuffle(candidates)
        samples.loc[candidates[: target_locked - current_locked], "evaluation_set"] = "locked"

    return samples


def select_roi_scans(samples, roi_scan_count, random_seed):
    """ROI 주석 대상도 calibration/locked와 난이도 비율을 유지해 선정합니다."""
    if roi_scan_count < 1 or roi_scan_count > len(samples):
        raise ValueError("roi_scan_count must be inside the selected pose scan range")

    calibration_count = int(round(roi_scan_count * 0.60))
    locked_count = roi_scan_count - calibration_count
    selected = []

    for offset, (name, count) in enumerate([("calibration", calibration_count), ("locked", locked_count)]):
        subset = samples[samples["evaluation_set"] == name]

        if count:
            selected.append(
                balanced_sample(
                    subset,
                    count,
                    random_seed + offset,
                    "roi_difficulty_quartile",
                )
            )

    return pd.concat(selected).reset_index(drop=True)


def save_annotation_template(outfile, data, key_columns, force=False):
    """기존 사람 주석은 보존하고 같은 표본 계약인지 확인합니다."""
    outfile = Path(outfile)

    if outfile.exists() and not force:
        existing = pd.read_csv(outfile, dtype={"scan_id": str})
        missing_columns = set(data.columns) - set(existing.columns)

        if missing_columns:
            raise ValueError(f"{outfile} is missing columns: {sorted(missing_columns)}")

        expected_keys = pd.MultiIndex.from_frame(data[key_columns])
        existing_keys = pd.MultiIndex.from_frame(existing[key_columns])

        if existing_keys.has_duplicates or set(existing_keys) != set(expected_keys):
            raise ValueError(f"{outfile} does not match the current GT sample. Use --force to reset it")

        editable = {
            "visibility",
            "x",
            "y",
            "x0",
            "y0",
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "annotator",
            "revision",
            "notes",
        }
        metadata_columns = [column for column in data.columns if column not in editable]
        expected_metadata = data.sort_values(key_columns)[metadata_columns].reset_index(drop=True)
        existing_metadata = existing.sort_values(key_columns)[metadata_columns].reset_index(drop=True)

        if not expected_metadata.equals(existing_metadata):
            raise ValueError(f"{outfile} metadata differs from the current GT sample")

        print(f"Preserved existing annotations: {outfile}")
        return

    write_csv(outfile, data)


def annotation_contract_hash(manifest):
    """GT 대상, split, 이미지 경로를 하나의 재현 가능한 hash로 고정합니다."""
    columns = [
        "scan_id",
        "view_index",
        "image_file",
        "difficulty_quartile",
        "roi_difficulty_quartile",
        "evaluation_set",
        "annotate_pose",
        "annotate_roi",
    ]
    records = []

    for row in manifest.sort_values(["scan_id", "view_index"])[columns].itertuples(index=False):
        records.append(
            {
                "scan_id": str(row.scan_id),
                "view_index": int(row.view_index),
                "image_file": str(row.image_file),
                "difficulty_quartile": int(row.difficulty_quartile),
                "roi_difficulty_quartile": int(row.roi_difficulty_quartile),
                "evaluation_set": str(row.evaluation_set),
                "annotate_pose": bool(row.annotate_pose),
                "annotate_roi": bool(row.annotate_roi),
            }
        )

    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_ground_truth_provenance(manifest, provenance):
    """Manifest가 최초 생성한 calibration/locked 계약과 같은지 확인합니다."""
    expected_hash = provenance.get("annotation_contract_sha256")

    if not valid_sha256(expected_hash):
        raise ValueError("GT provenance annotation contract is missing or invalid")

    if annotation_contract_hash(manifest) != expected_hash:
        raise ValueError("Annotation manifest differs from the frozen GT provenance contract")


def prepare_ground_truth(
    dataset_file,
    data_directory,
    artifact_file,
    output_directory,
    sample_count=40,
    roi_scan_count=20,
    random_seed=42,
    force=False,
):
    """Native APS 이미지와 Pose/ROI 정답지 입력 양식을 생성합니다."""
    candidates = ground_truth_candidates(dataset_file, artifact_file)
    samples = balanced_sample(candidates, sample_count, random_seed)
    samples = assign_evaluation_set(samples, random_seed + 1)
    roi_scans = select_roi_scans(samples, roi_scan_count, random_seed + 2)
    roi_scan_ids = set(roi_scans["scan_id"])

    extended_count = max(1, roi_scan_count // 2)
    extended_scans = balanced_sample(
        roi_scans,
        extended_count,
        random_seed + 3,
        "roi_difficulty_quartile",
    )
    extended_scan_ids = set(extended_scans["scan_id"])
    even_views = {0, 2, 4, 6, 8, 10, 12, 14}
    odd_views = {1, 3, 5, 7, 9, 11, 13, 15}

    output_directory = Path(output_directory)
    image_directory = output_directory / "images"
    artifact_sha256 = file_sha256(artifact_file)

    roi_views = {}

    for position, row in enumerate(roi_scans.sort_values("scan_id").itertuples(index=False)):
        base_views = even_views if position % 2 == 0 else odd_views
        selected_views = set(base_views)

        if row.scan_id in extended_scan_ids:
            if base_views == even_views:
                selected_views.update({1, 7, 9, 15})
            else:
                selected_views.update({0, 6, 8, 14})

        roi_views[row.scan_id] = selected_views

    manifest_rows = []
    pose_rows = []
    roi_rows = []

    for current, row in enumerate(samples.itertuples(index=False), start=1):
        images, _ = adapter(Path(data_directory) / row.aps_path)

        for view in range(VIEW):
            relative_image = Path(row.evaluation_set) / f"{row.scan_id}_v{view:02d}.png"
            image_file = image_directory / relative_image
            image_file.parent.mkdir(parents=True, exist_ok=True)

            saved_image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE) if image_file.exists() else None

            if force or saved_image is None or saved_image.shape != (HEIGHT, WIDTH):
                if not cv2.imwrite(str(image_file), images[view, :, :, 0]):
                    raise OSError(f"Failed to save {image_file}")

            annotate_roi = row.scan_id in roi_scan_ids and view in roi_views[row.scan_id]
            manifest_rows.append(
                {
                    "scan_id": row.scan_id,
                    "view_index": view,
                    "image_file": str(Path("images") / relative_image),
                    "difficulty_quartile": int(row.difficulty_quartile),
                    "roi_difficulty_quartile": int(row.roi_difficulty_quartile),
                    "evaluation_set": row.evaluation_set,
                    "annotate_pose": True,
                    "annotate_roi": bool(annotate_roi),
                }
            )

            for joint in POSE_JOINTS:
                pose_rows.append(
                    {
                        "scan_id": row.scan_id,
                        "view_index": view,
                        "joint_id": joint,
                        "joint_name": KEYPOINT_NAMES[joint],
                        "visibility": "",
                        "x": np.nan,
                        "y": np.nan,
                        "annotator": "",
                        "revision": 1,
                        "difficulty_quartile": int(row.difficulty_quartile),
                        "roi_difficulty_quartile": int(row.roi_difficulty_quartile),
                        "evaluation_set": row.evaluation_set,
                        "notes": "",
                    }
                )

            if annotate_roi:
                for zone in range(1, ZONE_COUNT + 1):
                    roi_rows.append(
                        {
                            "scan_id": row.scan_id,
                            "view_index": view,
                            "zone_id": zone,
                            "visibility": "",
                            **{f"x{point}": np.nan for point in range(4)},
                            **{f"y{point}": np.nan for point in range(4)},
                            "annotator": "",
                            "revision": 1,
                            "difficulty_quartile": int(row.difficulty_quartile),
                            "roi_difficulty_quartile": int(row.roi_difficulty_quartile),
                            "evaluation_set": row.evaluation_set,
                            "notes": "",
                        }
                    )

        print(f"[{current}/{len(samples)}] {row.scan_id}: Annotation images ready")

    manifest = pd.DataFrame(manifest_rows)
    pose_template = pd.DataFrame(pose_rows)
    roi_template = pd.DataFrame(roi_rows)

    with np.load(artifact_file, allow_pickle=False) as artifact:
        provenance = {
            "artifact_version": str(artifact["artifact_version"]),
            "pipeline_signature": str(artifact["pipeline_signature"]),
            "artifact_sha256": artifact_sha256,
            "annotation_contract_sha256": annotation_contract_hash(manifest),
            "coordinate_system": {"height": HEIGHT, "width": WIDTH, "origin": "top_left"},
            "annotation_image_preprocessing": "APS percentile 99.5 normalization followed by CLAHE grayscale",
            "roi_polygon_order": "four convex perimeter vertices in clockwise or counter-clockwise order",
            "pose_joints": {str(joint): KEYPOINT_NAMES[joint] for joint in POSE_JOINTS},
            "visibility_codes": {
                "V": "directly visible",
                "I": "inferable within approximately 12 pixels",
                "N": "not visible",
                "O": "outside frame",
            },
            "sample_count": len(samples),
            "roi_scan_count": len(roi_scans),
            "random_seed": random_seed,
        }

    provenance_file = output_directory / "provenance.json"

    if provenance_file.exists() and not force:
        with open(provenance_file, encoding="utf-8") as file:
            existing_provenance = json.load(file)

        if existing_provenance != provenance:
            raise ValueError("Existing GT provenance differs. Use --force only when annotations may be reset")

    save_annotation_template(
        output_directory / "annotation_manifest.csv",
        manifest,
        ["scan_id", "view_index"],
        force,
    )
    save_annotation_template(
        output_directory / "pose_ground_truth.csv",
        pose_template,
        ["scan_id", "view_index", "joint_id"],
        force,
    )
    save_annotation_template(
        output_directory / "roi_ground_truth.csv",
        roi_template,
        ["scan_id", "view_index", "zone_id"],
        force,
    )
    write_json(provenance_file, provenance)
    print(f"Prepared GT set: {output_directory}")


def load_pose_ground_truth(infile):
    """완료된 Pose GT의 좌표, 가시성, 중복 여부를 검사합니다."""
    data = pd.read_csv(infile, dtype={"scan_id": str})
    required = {
        "scan_id",
        "view_index",
        "joint_id",
        "joint_name",
        "visibility",
        "x",
        "y",
        "annotator",
        "evaluation_set",
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Pose GT is missing columns: {sorted(missing)}")

    keys = ["scan_id", "view_index", "joint_id"]

    if data.duplicated(keys).any():
        raise ValueError("Pose GT contains duplicate scan/view/joint rows")

    data["visibility"] = data["visibility"].fillna("").str.upper()

    if not data["visibility"].isin(["V", "I", "N", "O"]).all():
        count = int((~data["visibility"].isin(["V", "I", "N", "O"])).sum())
        raise ValueError(f"Pose GT has {count} incomplete or unknown visibility values")

    if not np.equal(data["joint_id"], data["joint_id"].astype(int)).all():
        raise ValueError("Pose GT joint_id must be an integer")

    if not data["joint_id"].isin(POSE_JOINTS).all():
        raise ValueError("Pose GT contains a joint outside the 12-joint contract")

    if not np.equal(data["view_index"], data["view_index"].astype(int)).all():
        raise ValueError("Pose GT view_index must be an integer")

    if not data["view_index"].between(0, VIEW - 1).all():
        raise ValueError("Pose GT contains a view outside 0..15")

    if not data["evaluation_set"].isin(["calibration", "locked"]).all():
        raise ValueError("Pose GT evaluation_set must be calibration or locked")

    visible = data["visibility"].isin(["V", "I"])
    coordinates = data[["x", "y"]].apply(pd.to_numeric, errors="coerce")

    if coordinates[visible].isna().any(axis=None):
        raise ValueError("Visible or inferable Pose GT requires x and y")

    if coordinates[~visible].notna().any(axis=None):
        raise ValueError("Not-visible or outside Pose GT must not contain coordinates")

    if not coordinates.loc[visible, "x"].between(0, WIDTH, inclusive="left").all():
        raise ValueError("Pose GT x coordinate is outside the native image")

    if not coordinates.loc[visible, "y"].between(0, HEIGHT, inclusive="left").all():
        raise ValueError("Pose GT y coordinate is outside the native image")

    if data["annotator"].fillna("").str.strip().eq("").any():
        raise ValueError("Every Pose GT row requires an annotator")

    data[["x", "y"]] = coordinates
    return data


def load_roi_ground_truth(infile):
    """완료된 ROI GT의 polygon, 가시성, 중복 여부를 검사합니다."""
    data = pd.read_csv(infile, dtype={"scan_id": str})
    coordinate_columns = [value for point in range(4) for value in (f"x{point}", f"y{point}")]
    required = {
        "scan_id",
        "view_index",
        "zone_id",
        "visibility",
        "annotator",
        "evaluation_set",
        *coordinate_columns,
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"ROI GT is missing columns: {sorted(missing)}")

    keys = ["scan_id", "view_index", "zone_id"]

    if data.duplicated(keys).any():
        raise ValueError("ROI GT contains duplicate scan/view/zone rows")

    data["visibility"] = data["visibility"].fillna("").str.upper()

    if not data["visibility"].isin(["V", "I", "N", "O"]).all():
        count = int((~data["visibility"].isin(["V", "I", "N", "O"])).sum())
        raise ValueError(f"ROI GT has {count} incomplete or unknown visibility values")

    if not np.equal(data["view_index"], data["view_index"].astype(int)).all():
        raise ValueError("ROI GT view_index must be an integer")

    if not data["view_index"].between(0, VIEW - 1).all():
        raise ValueError("ROI GT contains a view outside 0..15")

    if not np.equal(data["zone_id"], data["zone_id"].astype(int)).all():
        raise ValueError("ROI GT zone_id must be an integer")

    if not data["zone_id"].between(1, ZONE_COUNT).all():
        raise ValueError("ROI GT contains a zone outside 1..17")

    if not data["evaluation_set"].isin(["calibration", "locked"]).all():
        raise ValueError("ROI GT evaluation_set must be calibration or locked")

    visible = data["visibility"].isin(["V", "I"])
    coordinates = data[coordinate_columns].apply(pd.to_numeric, errors="coerce")

    if coordinates[visible].isna().any(axis=None):
        raise ValueError("Visible or inferable ROI GT requires four polygon points")

    if coordinates[~visible].notna().any(axis=None):
        raise ValueError("Not-visible or outside ROI GT must not contain polygon points")

    if data["annotator"].fillna("").str.strip().eq("").any():
        raise ValueError("Every ROI GT row requires an annotator")

    for row in coordinates[visible].itertuples(index=False, name=None):
        polygon = np.asarray(row, dtype=np.float32).reshape(4, 2)
        points = np.round(polygon).astype(np.int32)
        edges = np.roll(polygon, -1, axis=0) - polygon

        if np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] >= WIDTH):
            raise ValueError("ROI GT x coordinate is outside the native image")

        if np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] >= HEIGHT):
            raise ValueError("ROI GT y coordinate is outside the native image")

        if abs(float(cv2.contourArea(points))) < 50:
            raise ValueError("ROI GT polygon area must be at least 50 pixels")

        if np.linalg.norm(edges, axis=1).min() < 4 or not cv2.isContourConvex(points):
            raise ValueError("ROI GT polygon must be a non-degenerate convex quadrilateral")

    data[coordinate_columns] = coordinates
    return data


def load_annotation_manifest(infile, ground_truth_directory):
    """GT manifest의 이미지, split, 주석 대상 계약을 검사합니다."""
    data = pd.read_csv(infile, dtype={"scan_id": str})
    required = {
        "scan_id",
        "view_index",
        "image_file",
        "evaluation_set",
        "annotate_pose",
        "annotate_roi",
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Annotation manifest is missing columns: {sorted(missing)}")

    if data.duplicated(["scan_id", "view_index"]).any():
        raise ValueError("Annotation manifest contains duplicate scan/view rows")

    if not np.equal(data["view_index"], data["view_index"].astype(int)).all():
        raise ValueError("Annotation manifest view_index must be an integer")

    if not data["view_index"].between(0, VIEW - 1).all():
        raise ValueError("Annotation manifest contains a view outside 0..15")

    if not data["evaluation_set"].isin(["calibration", "locked"]).all():
        raise ValueError("Annotation manifest evaluation_set must be calibration or locked")

    if not data.groupby("scan_id")["evaluation_set"].nunique().eq(1).all():
        raise ValueError("A scan cannot appear in both calibration and locked sets")

    view_sets = data.groupby("scan_id")["view_index"].apply(lambda values: set(values.astype(int)))

    if not view_sets.map(lambda values: values == set(range(VIEW))).all():
        raise ValueError("Every annotation scan must contain views 0..15 exactly once")

    for column in ["annotate_pose", "annotate_roi"]:
        values = data[column].astype(str).str.lower().map({"true": True, "false": False})

        if values.isna().any():
            raise ValueError(f"Annotation manifest {column} must be boolean")

        data[column] = values.astype(bool)

    if not data["annotate_pose"].all():
        raise ValueError("Every annotation image must be included in Pose GT")

    if data["image_file"].duplicated().any():
        raise ValueError("Annotation manifest contains duplicate image paths")

    expected_images = data.apply(
        lambda row: str(Path("images") / row.evaluation_set / f"{row.scan_id}_v{int(row.view_index):02d}.png"),
        axis=1,
    )

    if not (data["image_file"] == expected_images).all():
        raise ValueError("Annotation image path does not match its scan, view, and split")

    missing_images = [
        image_file for image_file in data["image_file"] if not (Path(ground_truth_directory) / image_file).is_file()
    ]

    if missing_images:
        raise ValueError(f"Annotation images are missing: {missing_images[:5]}")

    return data


def validate_ground_truth_coverage(manifest, pose_ground_truth, roi_ground_truth):
    """GT 행 삭제나 split 변경 없이 manifest의 모든 대상을 주석했는지 확인합니다."""
    pose_manifest = manifest[manifest["annotate_pose"]][["scan_id", "view_index", "evaluation_set"]].copy()
    joints = pd.DataFrame(
        {
            "joint_id": POSE_JOINTS,
            "joint_name": [KEYPOINT_NAMES[joint] for joint in POSE_JOINTS],
        }
    )
    pose_expected = pose_manifest.merge(joints, how="cross")
    pose_keys = ["scan_id", "view_index", "joint_id"]
    expected_pose_keys = pd.MultiIndex.from_frame(pose_expected[pose_keys])
    actual_pose_keys = pd.MultiIndex.from_frame(pose_ground_truth[pose_keys])

    if set(expected_pose_keys) != set(actual_pose_keys):
        raise ValueError("Pose GT rows do not exactly match annotation_manifest.csv")

    pose_check = pose_ground_truth.merge(
        pose_expected,
        on=pose_keys,
        suffixes=("", "_expected"),
    )

    if not (pose_check["evaluation_set"] == pose_check["evaluation_set_expected"]).all():
        raise ValueError("Pose GT evaluation_set differs from annotation manifest")

    if not (pose_check["joint_name"] == pose_check["joint_name_expected"]).all():
        raise ValueError("Pose GT joint_name differs from joint_id")

    roi_manifest = manifest[manifest["annotate_roi"]][["scan_id", "view_index", "evaluation_set"]].copy()
    zones = pd.DataFrame({"zone_id": np.arange(1, ZONE_COUNT + 1)})
    roi_expected = roi_manifest.merge(zones, how="cross")
    roi_keys = ["scan_id", "view_index", "zone_id"]
    expected_roi_keys = pd.MultiIndex.from_frame(roi_expected[roi_keys])
    actual_roi_keys = pd.MultiIndex.from_frame(roi_ground_truth[roi_keys])

    if set(expected_roi_keys) != set(actual_roi_keys):
        raise ValueError("ROI GT rows do not exactly match annotation_manifest.csv")

    roi_check = roi_ground_truth.merge(
        roi_expected,
        on=roi_keys,
        suffixes=("", "_expected"),
    )

    if not (roi_check["evaluation_set"] == roi_check["evaluation_set_expected"]).all():
        raise ValueError("ROI GT evaluation_set differs from annotation manifest")


def prepare_pose_training_data(
    dataset_file,
    artifact_file,
    ground_truth_directory,
    output_directory,
    random_seed=42,
    force=False,
):
    """Calibration Pose GT를 Ultralytics 12-keypoint 학습 형식으로 변환합니다."""
    ground_truth_directory = Path(ground_truth_directory)
    manifest = load_annotation_manifest(
        ground_truth_directory / "annotation_manifest.csv",
        ground_truth_directory,
    )
    pose_ground_truth = load_pose_ground_truth(ground_truth_directory / "pose_ground_truth.csv")
    roi_ground_truth = load_roi_ground_truth(ground_truth_directory / "roi_ground_truth.csv")
    validate_ground_truth_coverage(manifest, pose_ground_truth, roi_ground_truth)

    with open(ground_truth_directory / "provenance.json", encoding="utf-8") as file:
        provenance = json.load(file)

    validate_ground_truth_provenance(manifest, provenance)

    if provenance.get("artifact_sha256") != file_sha256(artifact_file):
        raise ValueError("Pose training artifact differs from GT provenance")

    output_directory = Path(output_directory)

    if output_directory.exists() and any(output_directory.iterdir()):
        if not force:
            raise ValueError(f"{output_directory} is not empty. Use --force to rebuild generated training data")

        for name in ["images", "labels"]:
            directory = output_directory / name

            if directory.exists():
                shutil.rmtree(directory)

    calibration = manifest[manifest["evaluation_set"] == "calibration"]
    scans = calibration[["scan_id", "difficulty_quartile"]].drop_duplicates().reset_index(drop=True)

    if len(scans) < 2:
        raise ValueError("At least two calibration scans are required for pose training")

    validation_count = max(1, int(round(len(scans) * 0.20)))
    validation_scans = balanced_sample(scans, validation_count, random_seed)
    validation_ids = set(validation_scans["scan_id"])
    pose_by_frame = {key: rows for key, rows in pose_ground_truth.groupby(["scan_id", "view_index"])}
    training_rows = []

    with np.load(artifact_file, allow_pickle=False) as artifact:
        errors = validate_aggregate(artifact)

        if errors:
            raise ValueError("Invalid artifact: " + "; ".join(errors))

        scan_index = {scan_id: index for index, scan_id in enumerate(artifact["scan_ids"].astype(str))}

        for row in calibration.itertuples(index=False):
            split = "val" if row.scan_id in validation_ids else "train"
            source_image = ground_truth_directory / row.image_file
            image_file = output_directory / "images" / split / source_image.name
            label_file = output_directory / "labels" / split / f"{source_image.stem}.txt"
            image_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, image_file)

            points = pose_by_frame[(row.scan_id, row.view_index)].set_index("joint_id")
            visible_points = points[points["visibility"].isin(["V", "I"])][["x", "y"]].to_numpy()

            if len(visible_points) < 4:
                label_file.unlink(missing_ok=True)
                image_file.unlink(missing_ok=True)
                continue

            index = scan_index[row.scan_id]
            body_box = artifact["body_boxes"][index, row.view_index].astype(float)

            if np.any(body_box < 0):
                minimum = visible_points.min(axis=0)
                maximum = visible_points.max(axis=0)
                margin = (maximum - minimum) * 0.10
                body_box = clip_box((*tuple(minimum - margin), *tuple(maximum + margin))).astype(float)

            x1, y1, x2, y2 = body_box
            values = [
                "0",
                f"{(x1 + x2) / (2 * WIDTH):.6f}",
                f"{(y1 + y2) / (2 * HEIGHT):.6f}",
                f"{(x2 - x1) / WIDTH:.6f}",
                f"{(y2 - y1) / HEIGHT:.6f}",
            ]

            for joint in POSE_JOINTS:
                joint_row = points.loc[joint]

                if joint_row.visibility in {"V", "I"}:
                    visibility = 2 if joint_row.visibility == "V" else 1
                    values.extend(
                        [
                            f"{joint_row.x / WIDTH:.6f}",
                            f"{joint_row.y / HEIGHT:.6f}",
                            str(visibility),
                        ]
                    )
                else:
                    values.extend(["0.000000", "0.000000", "0"])

            write_text(label_file, " ".join(values) + "\n")
            training_rows.append(
                {
                    "scan_id": row.scan_id,
                    "view_index": int(row.view_index),
                    "split": split,
                    "image_file": str(image_file.relative_to(output_directory)),
                    "label_file": str(label_file.relative_to(output_directory)),
                    "visible_keypoints": len(visible_points),
                }
            )

    flip_indices = []

    for joint in POSE_JOINTS:
        pair = next((pair for pair in SYMMETRIC_JOINTS if joint in pair), None)
        opposite = pair[1] if pair and pair[0] == joint else pair[0]
        flip_indices.append(POSE_JOINTS.index(opposite))

    yaml = f"""path: {output_directory.resolve()}
train: images/train
val: images/val
kpt_shape: [12, 3]
flip_idx: {flip_indices}
names:
  0: person
"""
    write_text(output_directory / "pose_dataset.yaml", yaml)
    write_csv(output_directory / "training_manifest.csv", pd.DataFrame(training_rows))

    dataset = pd.read_csv(dataset_file, usecols=["scan_id", "aps_path", "type"])
    evaluation_ids = set(manifest["scan_id"].astype(str))
    evaluation_dataset = dataset[dataset["scan_id"].astype(str).isin(evaluation_ids)]
    write_csv(output_directory / "evaluation_dataset.csv", evaluation_dataset)
    write_json(
        output_directory / "training_provenance.json",
        {
            "source_artifact_sha256": file_sha256(artifact_file),
            "ground_truth_provenance": provenance,
            "joint_order": [KEYPOINT_NAMES[joint] for joint in POSE_JOINTS],
            "train_scan_count": len(scans) - len(validation_ids),
            "validation_scan_count": len(validation_ids),
            "locked_scans_excluded": sorted(manifest.loc[manifest["evaluation_set"] == "locked", "scan_id"].unique()),
            "random_seed": random_seed,
        },
    )
    print(f"Prepared Pose training data: {output_directory}")


def pose_ground_truth_scales(ground_truth):
    """Shoulder와 Hip 중심 사이의 길이를 normalized error 기준으로 계산합니다."""
    scales = {}
    visible = ground_truth[ground_truth["visibility"].isin(["V", "I"])]

    for (scan_id, view), rows in visible.groupby(["scan_id", "view_index"]):
        points = {int(row.joint_id): np.array([row.x, row.y]) for row in rows.itertuples()}
        required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

        if all(joint in points for joint in required):
            shoulder = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
            hip = midpoint(points[LEFT_HIP], points[RIGHT_HIP])
            scale = float(np.linalg.norm(shoulder - hip))

            if scale >= 20:
                scales[(scan_id, int(view))] = scale

    scan_scales = {}

    for scan_id in visible["scan_id"].unique():
        values = [value for (current, _), value in scales.items() if current == scan_id]

        if values:
            scan_scales[scan_id] = float(np.median(values))

    return scales, scan_scales


def pose_error_table(artifact, ground_truth):
    """Raw, label-only, full-refined Pose를 같은 GT 좌표에서 비교합니다."""
    scan_ids = artifact["scan_ids"].astype(str)
    scan_index = {scan_id: index for index, scan_id in enumerate(scan_ids)}
    unknown = sorted(set(ground_truth["scan_id"]) - set(scan_index))

    if unknown:
        raise ValueError(f"Pose GT scans are missing from artifact: {unknown[:5]}")

    scales, scan_scales = pose_ground_truth_scales(ground_truth)
    aligned_cache = {}
    rows = []

    for row in ground_truth.itertuples(index=False):
        index = scan_index[row.scan_id]
        view = int(row.view_index)
        joint = int(row.joint_id)
        visible = row.visibility in {"V", "I"}

        if row.scan_id not in aligned_cache:
            aligned_cache[row.scan_id] = align_pose_labels(
                artifact["raw_keypoints"][index],
                artifact["raw_keypoint_confidence"][index],
                artifact["frame_confidence"][index],
            )

        aligned_keypoints, aligned_confidence, aligned_source, _, aligned_valid = aligned_cache[row.scan_id]
        predictions = {
            "raw": (
                artifact["raw_keypoints"][index, view, joint],
                float(artifact["raw_keypoint_confidence"][index, view, joint]),
                False,
                KEYPOINT_OBSERVED,
            ),
            "aligned": (
                aligned_keypoints[view, joint],
                float(aligned_confidence[view, joint]),
                bool(aligned_valid[view, joint]),
                int(aligned_source[view, joint]),
            ),
            "refined": (
                artifact["keypoints"][index, view, joint],
                float(artifact["keypoint_confidence"][index, view, joint]),
                bool(artifact["keypoint_valid"][index, view, joint]),
                int(artifact["keypoint_source"][index, view, joint]),
            ),
        }
        gt_point = np.array([row.x, row.y], dtype=float) if visible else None
        scale = scales.get((row.scan_id, view), scan_scales.get(row.scan_id, np.nan))

        for model, (point, confidence, valid, source) in predictions.items():
            if model == "raw":
                valid = confidence >= MINIMUM_POSE_CONFIDENCE and valid_point(point)

            error = float(np.linalg.norm(point - gt_point)) if visible and valid else np.nan
            rows.append(
                {
                    "scan_id": row.scan_id,
                    "view_index": view,
                    "joint_id": joint,
                    "joint_name": KEYPOINT_NAMES[joint],
                    "joint_group": POSE_JOINT_GROUPS[joint],
                    "visibility": row.visibility,
                    "evaluation_set": row.evaluation_set,
                    "model": model,
                    "predicted_valid": bool(valid),
                    "confidence": confidence,
                    "source": source,
                    "correction_flags": int(artifact["correction_flags"][index, view]),
                    "predicted_x": float(point[0]),
                    "predicted_y": float(point[1]),
                    "gt_x": float(row.x) if visible else np.nan,
                    "gt_y": float(row.y) if visible else np.nan,
                    "error_pixels": error,
                    "normalized_error": error / scale if np.isfinite(error) and np.isfinite(scale) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize_pose_rows(data, model, evaluation_set, visibility, group="all"):
    """Pose 후보 하나의 coverage와 거리 기반 지표를 요약합니다."""
    subset = data[data["model"] == model]

    if evaluation_set != "all":
        subset = subset[subset["evaluation_set"] == evaluation_set]

    subset = subset[subset["visibility"].isin(visibility)]

    if group != "all":
        subset = subset[subset["joint_group"] == group]

    errors = subset["error_pixels"].dropna().to_numpy()
    normalized = subset["normalized_error"].dropna().to_numpy()
    total = len(subset)
    missing = int(subset["error_pixels"].isna().sum())

    return {
        "model": model,
        "evaluation_set": evaluation_set,
        "visibility_scope": "+".join(visibility),
        "joint_group": group,
        "ground_truth_points": total,
        "predicted_points": len(errors),
        "coverage": float(len(errors) / total) if total else 0.0,
        "mean_error": float(errors.mean()) if len(errors) else None,
        "median_error": float(np.median(errors)) if len(errors) else None,
        "p90_error": float(np.percentile(errors, 90)) if len(errors) else None,
        "median_normalized_error": float(np.median(normalized)) if len(normalized) else None,
        "pck_10": float(((subset["error_pixels"] <= 10) & subset["error_pixels"].notna()).mean()) if total else 0.0,
        "pck_20": float(((subset["error_pixels"] <= 20) & subset["error_pixels"].notna()).mean()) if total else 0.0,
        "pck_30": float(((subset["error_pixels"] <= 30) & subset["error_pixels"].notna()).mean()) if total else 0.0,
        "catastrophic_30": (
            float(((subset["error_pixels"] > 30) | subset["error_pixels"].isna()).mean()) if total else 0.0
        ),
        "catastrophic_50": (
            float(((subset["error_pixels"] > 50) | subset["error_pixels"].isna()).mean()) if total else 0.0
        ),
        "missing_predictions": missing,
    }


def pose_summary_table(data):
    """평가 split, 가시성, joint family별 Pose 지표를 생성합니다."""
    rows = []

    for model in ["raw", "aligned", "refined"]:
        for evaluation_set in ["all", "calibration", "locked"]:
            for visibility in [["V"], ["V", "I"]]:
                for group in ["all", *sorted(set(POSE_JOINT_GROUPS.values()))]:
                    rows.append(summarize_pose_rows(data, model, evaluation_set, visibility, group))

    return pd.DataFrame(rows)


def pose_action_summary(data):
    """각 보정 action이 실제 GT 오차를 줄였는지 point 단위로 검사합니다."""
    keys = ["scan_id", "view_index", "joint_id"]
    raw = data[data["model"] == "raw"][keys + ["error_pixels"]].rename(columns={"error_pixels": "error_before"})
    refined = data[
        (data["model"] == "refined") & (data["evaluation_set"] == "locked") & (data["visibility"] == "V")
    ].merge(raw, on=keys)
    actions = []

    for row in refined.itertuples(index=False):
        action = None

        if row.source == KEYPOINT_SWAPPED:
            if row.correction_flags & ROTATION_CORRECTION:
                action = "rotation_swap"
            elif row.correction_flags & LOWER_BODY_CORRECTION:
                action = "lower_body_swap"
        elif row.source == KEYPOINT_ROTATION:
            action = "structural_replacement"
        elif row.source == KEYPOINT_ARM:
            action = "arm_replacement"
        elif row.source == KEYPOINT_INTERPOLATED:
            action = "interpolation"

        if action is None or not np.isfinite(row.error_before) or not np.isfinite(row.error_pixels):
            continue

        actions.append(
            {
                "scan_id": row.scan_id,
                "view_index": row.view_index,
                "joint_id": row.joint_id,
                "action": action,
                "error_before": row.error_before,
                "error_after": row.error_pixels,
                "improvement": row.error_before - row.error_pixels,
            }
        )

    action_data = pd.DataFrame(actions)
    summaries = []

    if len(action_data):
        for action, rows in action_data.groupby("action"):
            summaries.append(
                {
                    "action": action,
                    "point_count": len(rows),
                    "scan_count": rows["scan_id"].nunique(),
                    "median_improvement": float(rows["improvement"].median()),
                    "improved_point_rate": float((rows["improvement"] > 0).mean()),
                    "action_gate": bool(
                        len(rows) >= 20
                        and rows["improvement"].median() > 0
                        and (rows["improvement"] > 0).mean() >= 0.65
                    ),
                }
            )

    columns = [
        "action",
        "point_count",
        "scan_count",
        "median_improvement",
        "improved_point_rate",
        "action_gate",
    ]
    return action_data, pd.DataFrame(summaries, columns=columns)


def paired_pose_bootstrap(data, first, second, iterations=2000, random_seed=42):
    """scan을 재표집하여 두 Pose 후보의 median error 차이 CI를 계산합니다."""
    first_data = data[data["model"] == first]
    second_data = data[data["model"] == second]
    keys = ["scan_id", "view_index", "joint_id"]
    paired = first_data.merge(second_data, on=keys, suffixes=("_first", "_second"))
    paired = paired[(paired["evaluation_set_first"] == "locked") & (paired["visibility_first"] == "V")]
    paired = paired.dropna(subset=["error_pixels_first", "error_pixels_second"])
    scan_ids = paired["scan_id"].unique()

    if len(scan_ids) < 2:
        return {"first": first, "second": second, "scan_count": len(scan_ids), "ci_low": None, "ci_high": None}

    by_scan = {scan_id: paired[paired["scan_id"] == scan_id] for scan_id in scan_ids}
    generator = np.random.default_rng(random_seed)
    differences = []

    for _ in range(iterations):
        sampled = generator.choice(scan_ids, len(scan_ids), replace=True)
        current = pd.concat([by_scan[scan_id] for scan_id in sampled], ignore_index=True)
        difference = current["error_pixels_second"].median() - current["error_pixels_first"].median()
        differences.append(float(difference))

    return {
        "first": first,
        "second": second,
        "scan_count": len(scan_ids),
        "median_difference": float(paired["error_pixels_second"].median() - paired["error_pixels_first"].median()),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
    }


def pose_swap_rates(data):
    """GT anatomical left/right와 반대 배치된 관절 쌍의 비율을 계산합니다."""
    pairs = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_ELBOW, RIGHT_ELBOW),
        (LEFT_WRIST, RIGHT_WRIST),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_KNEE, RIGHT_KNEE),
        (LEFT_ANKLE, RIGHT_ANKLE),
    ]
    results = []

    for model in ["raw", "aligned", "refined"]:
        subset = data[
            (data["model"] == model)
            & (data["evaluation_set"] == "locked")
            & (data["visibility"] == "V")
            & data["predicted_valid"]
        ]
        trials = 0
        swaps = 0

        for (_, _), frame in subset.groupby(["scan_id", "view_index"]):
            by_joint = {int(row.joint_id): row for row in frame.itertuples()}

            for left, right in pairs:
                if left not in by_joint or right not in by_joint:
                    continue

                left_row = by_joint[left]
                right_row = by_joint[right]
                predicted_left = np.array([left_row.predicted_x, left_row.predicted_y])
                predicted_right = np.array([right_row.predicted_x, right_row.predicted_y])
                gt_left = np.array([left_row.gt_x, left_row.gt_y])
                gt_right = np.array([right_row.gt_x, right_row.gt_y])
                assigned = np.linalg.norm(predicted_left - gt_left) + np.linalg.norm(predicted_right - gt_right)
                swapped = np.linalg.norm(predicted_left - gt_right) + np.linalg.norm(predicted_right - gt_left)
                trials += 1
                swaps += int(swapped + 5 < assigned)

        results.append(
            {
                "model": model,
                "pair_count": trials,
                "swap_count": swaps,
                "swap_rate": swaps / trials if trials else None,
            }
        )

    return results


def oriented_limb_candidate(keypoints, confidence, zone):
    """명암 Profile을 쓰지 않은 Pose 기반 limb counterfactual을 계산합니다."""
    start_joint, end_joint, _, _ = JOINTS[zone]

    if confidence[[start_joint, end_joint]].min() < MINIMUM_POSE_CONFIDENCE:
        return None

    start, end = segmentor(keypoints, zone)
    direction = end - start
    length = float(np.linalg.norm(direction))

    if not valid_point(start) or not valid_point(end) or length < LIMB_MINIMUM_LENGTH:
        return None

    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]])
    half_width = expected_limb_width(zone, length) / 2
    start = start - tangent * length * 0.05
    end = end + tangent * length * 0.05
    polygon = np.array(
        [
            start - normal * half_width,
            start + normal * half_width,
            end + normal * half_width,
            end - normal * half_width,
        ],
        dtype=np.float32,
    )
    polygon, clipping_ratio = clip_polygon(polygon)
    points = np.round(polygon).astype(np.int32)
    edges = np.roll(polygon, -1, axis=0) - polygon

    if clipping_ratio < 0.70:
        return None

    if abs(float(cv2.contourArea(points))) < 50:
        return None

    if np.linalg.norm(edges, axis=1).min() < 4 or not cv2.isContourConvex(points):
        return None

    return polygon


def sample_polygon_boundary(polygon, spacing=4):
    """Polygon edge를 일정 간격의 점으로 바꿉니다."""
    samples = []

    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        count = max(2, int(np.linalg.norm(end - start) / spacing) + 1)
        samples.append(np.linspace(start, end, count, endpoint=False))

    return np.concatenate(samples)


def polygon_overlap_metrics(prediction, ground_truth):
    """두 convex polygon의 IoU, coverage, precision, 위치 오차를 계산합니다."""
    prediction = np.asarray(prediction, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    prediction_area = abs(float(cv2.contourArea(prediction)))
    ground_truth_area = abs(float(cv2.contourArea(ground_truth)))
    intersection, _ = cv2.intersectConvexConvex(prediction, ground_truth)
    intersection = float(intersection)
    union = prediction_area + ground_truth_area - intersection
    prediction_boundary = sample_polygon_boundary(prediction)
    ground_truth_boundary = sample_polygon_boundary(ground_truth)
    distance = np.linalg.norm(
        prediction_boundary[:, None, :] - ground_truth_boundary[None, :, :],
        axis=2,
    )
    boundary_distance = (distance.min(axis=0).mean() + distance.min(axis=1).mean()) / 2
    ground_truth_box = polygon_box(ground_truth).astype(float)
    diagonal = float(np.linalg.norm(ground_truth_box[2:] - ground_truth_box[:2]))
    iou = float(np.clip(intersection / max(union, 1e-6), 0, 1))
    dice = float(np.clip(2 * intersection / max(prediction_area + ground_truth_area, 1e-6), 0, 1))
    coverage = float(np.clip(intersection / max(ground_truth_area, 1e-6), 0, 1))
    precision = float(np.clip(intersection / max(prediction_area, 1e-6), 0, 1))

    return {
        "iou": iou,
        "dice": dice,
        "coverage": coverage,
        "precision": precision,
        "background_excess": 1 - precision,
        "centroid_distance": float(np.linalg.norm(prediction.mean(axis=0) - ground_truth.mean(axis=0))),
        "normalized_centroid_distance": float(
            np.linalg.norm(prediction.mean(axis=0) - ground_truth.mean(axis=0)) / max(diagonal, 1e-6)
        ),
        "boundary_distance": float(boundary_distance),
    }


def roi_error_table(artifact, ground_truth):
    """선택 ROI와 oriented/relative counterfactual을 동일 GT에서 비교합니다."""
    scan_ids = artifact["scan_ids"].astype(str)
    scan_index = {scan_id: index for index, scan_id in enumerate(scan_ids)}
    unknown = sorted(set(ground_truth["scan_id"]) - set(scan_index))

    if unknown:
        raise ValueError(f"ROI GT scans are missing from artifact: {unknown[:5]}")

    source_names = {
        ROI_INVALID: "invalid",
        ROI_INTENSITY: "intensity",
        ROI_ORIENTED: "oriented",
        ROI_TORSO: "torso",
        ROI_RELATIVE: "relative",
    }
    rows = []

    for row in ground_truth.itertuples(index=False):
        index = scan_index[row.scan_id]
        view = int(row.view_index)
        zone = int(row.zone_id)
        zone_index = zone - 1
        gt_visible = row.visibility in {"V", "I"}
        ground_truth_polygon = None

        if gt_visible:
            ground_truth_polygon = np.array(
                [[getattr(row, f"x{point}"), getattr(row, f"y{point}")] for point in range(4)],
                dtype=np.float32,
            )

        selected_valid = bool(artifact["roi_valid"][index, view, zone_index])
        selected_polygon = artifact["polygons"][index, view, zone_index] if selected_valid else None
        selected_source = int(artifact["roi_source"][index, view, zone_index])
        oriented_polygon = None

        if zone in JOINTS:
            oriented_polygon = oriented_limb_candidate(
                artifact["keypoints"][index, view],
                artifact["keypoint_confidence"][index, view],
                zone,
            )

        relative = relative_polygon(artifact["body_boxes"][index, view], zone, view)
        relative_candidate = relative[0] if relative is not None else None
        candidates = [
            (
                "selected",
                selected_polygon,
                selected_valid,
                bool(artifact["roi_visible"][index, view, zone_index]),
                selected_source,
                float(artifact["roi_quality"][index, view, zone_index]),
            ),
            (
                "oriented_counterfactual",
                oriented_polygon,
                oriented_polygon is not None,
                oriented_polygon is not None,
                ROI_ORIENTED,
                0.0,
            ),
            (
                "relative_counterfactual",
                relative_candidate,
                relative_candidate is not None,
                relative_candidate is not None,
                ROI_RELATIVE,
                0.0,
            ),
        ]

        for candidate, polygon, predicted_valid, predicted_visible, source, quality in candidates:
            metrics = {}

            if gt_visible and predicted_valid:
                metrics = polygon_overlap_metrics(polygon, ground_truth_polygon)

            rows.append(
                {
                    "scan_id": row.scan_id,
                    "view_index": view,
                    "zone_id": zone,
                    "zone_group": ROI_ZONE_GROUPS[zone],
                    "visibility": row.visibility,
                    "evaluation_set": row.evaluation_set,
                    "candidate": candidate,
                    "predicted_valid": bool(predicted_valid),
                    "predicted_visible": bool(predicted_visible),
                    "source": source_names[source],
                    "quality": quality,
                    **{
                        key: metrics.get(key, np.nan)
                        for key in [
                            "iou",
                            "dice",
                            "coverage",
                            "precision",
                            "background_excess",
                            "centroid_distance",
                            "normalized_centroid_distance",
                            "boundary_distance",
                        ]
                    },
                }
            )

    return pd.DataFrame(rows)


def summarize_roi_rows(data, candidate, evaluation_set, visibility, group="all", source="all"):
    """ROI 후보 하나의 overlap과 catastrophic rate를 요약합니다."""
    subset = data[data["candidate"] == candidate]

    if evaluation_set != "all":
        subset = subset[subset["evaluation_set"] == evaluation_set]

    subset = subset[subset["visibility"].isin(visibility)]

    if group != "all":
        subset = subset[subset["zone_group"] == group]

    if source != "all":
        subset = subset[subset["source"] == source]

    iou = subset["iou"].dropna().to_numpy()
    coverage = subset["coverage"].dropna().to_numpy()
    precision = subset["precision"].dropna().to_numpy()
    total = len(subset)

    return {
        "candidate": candidate,
        "evaluation_set": evaluation_set,
        "visibility_scope": "+".join(visibility),
        "zone_group": group,
        "source": source,
        "ground_truth_rois": total,
        "predicted_rois": len(iou),
        "prediction_coverage": float(len(iou) / total) if total else 0.0,
        "median_iou": float(np.median(iou)) if len(iou) else None,
        "p25_iou": float(np.percentile(iou, 25)) if len(iou) else None,
        "median_coverage": float(np.median(coverage)) if len(coverage) else None,
        "median_precision": float(np.median(precision)) if len(precision) else None,
        "catastrophic_020": float(((subset["iou"] < 0.20) | subset["iou"].isna()).mean()) if total else 0.0,
    }


def roi_summary_table(data):
    """평가 split, 가시성, zone group, ROI source별 geometry 지표를 생성합니다."""
    rows = []

    for candidate in ["selected", "oriented_counterfactual", "relative_counterfactual"]:
        for evaluation_set in ["all", "calibration", "locked"]:
            for visibility in [["V"], ["V", "I"]]:
                for group in ["all", *sorted(set(ROI_ZONE_GROUPS.values()))]:
                    rows.append(summarize_roi_rows(data, candidate, evaluation_set, visibility, group))

                if candidate == "selected":
                    for source in ["intensity", "oriented", "torso", "relative"]:
                        rows.append(summarize_roi_rows(data, candidate, evaluation_set, visibility, source=source))

    return pd.DataFrame(rows)


def roi_quality_calibration(data):
    """Locked GT에서 roi_quality가 실제 IoU를 순서화하는지 검사합니다."""
    subset = data[(data["candidate"] == "selected") & (data["evaluation_set"] == "locked") & data["iou"].notna()].copy()

    if len(subset) < 2:
        return {"count": len(subset), "spearman": None, "auroc_iou_050": None}

    spearman = subset["quality"].rank().corr(subset["iou"].rank())
    positive = subset["iou"] >= 0.50
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    auroc = None

    if positive_count and negative_count:
        ranks = subset["quality"].rank(method="average")
        rank_sum = float(ranks[positive].sum())
        auroc = (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)

    return {
        "count": len(subset),
        "spearman": float(spearman) if np.isfinite(spearman) else None,
        "auroc_iou_050": float(auroc) if auroc is not None else None,
    }


def roi_scan_bootstrap(data, candidate="selected", iterations=2000, random_seed=42):
    """scan 단위 재표집으로 locked ROI median IoU의 신뢰구간을 계산합니다."""
    subset = data[(data["candidate"] == candidate) & (data["evaluation_set"] == "locked") & (data["visibility"] == "V")]
    scan_ids = subset["scan_id"].unique()

    if len(scan_ids) < 2:
        return {"candidate": candidate, "scan_count": len(scan_ids), "ci_low": None, "ci_high": None}

    by_scan = {scan_id: subset[subset["scan_id"] == scan_id] for scan_id in scan_ids}
    generator = np.random.default_rng(random_seed)
    medians = []

    for _ in range(iterations):
        sampled = generator.choice(scan_ids, len(scan_ids), replace=True)
        current = pd.concat([by_scan[scan_id] for scan_id in sampled], ignore_index=True)
        values = current["iou"].dropna()

        if len(values):
            medians.append(float(values.median()))

    return {
        "candidate": candidate,
        "scan_count": len(scan_ids),
        "median_iou": float(subset["iou"].median()) if subset["iou"].notna().any() else None,
        "ci_low": float(np.percentile(medians, 2.5)) if medians else None,
        "ci_high": float(np.percentile(medians, 97.5)) if medians else None,
    }


def select_summary_row(data, **values):
    """Summary table에서 지정한 조건의 한 행을 사전으로 반환합니다."""
    selected = data

    for column, value in values.items():
        selected = selected[selected[column] == value]

    if len(selected) != 1:
        return None

    return selected.iloc[0].to_dict()


def finite_metric(value):
    """Summary 결측값이 실제 유한한 수치인지 확인합니다."""
    return value is not None and np.isfinite(value)


def evaluation_sample_status(pose_ground_truth, roi_ground_truth):
    """Pilot 결과가 확정 gate로 오인되지 않도록 locked GT 규모를 검사합니다."""
    locked_pose = pose_ground_truth[
        (pose_ground_truth["evaluation_set"] == "locked") & (pose_ground_truth["visibility"] == "V")
    ]
    locked_roi = roi_ground_truth[
        (roi_ground_truth["evaluation_set"] == "locked") & roi_ground_truth["visibility"].isin(["V", "I"])
    ]
    pose_ready = locked_pose["scan_id"].nunique() >= MINIMUM_LOCKED_POSE_SCANS
    pose_ready &= len(locked_pose) >= MINIMUM_LOCKED_POSE_POINTS
    roi_ready = locked_roi["scan_id"].nunique() >= MINIMUM_LOCKED_ROI_SCANS
    roi_ready &= len(locked_roi) >= MINIMUM_LOCKED_ROIS

    return {
        "pilot_only": not (pose_ready and roi_ready),
        "pose_ready": bool(pose_ready),
        "roi_ready": bool(roi_ready),
        "locked_pose_scans": int(locked_pose["scan_id"].nunique()),
        "locked_pose_visible_points": len(locked_pose),
        "locked_roi_scans": int(locked_roi["scan_id"].nunique()),
        "locked_roi_visible_polygons": len(locked_roi),
        "minimum_locked_pose_scans": MINIMUM_LOCKED_POSE_SCANS,
        "minimum_locked_pose_points": MINIMUM_LOCKED_POSE_POINTS,
        "minimum_locked_roi_scans": MINIMUM_LOCKED_ROI_SCANS,
        "minimum_locked_rois": MINIMUM_LOCKED_ROIS,
    }


def pose_decisions(summary, swaps, bootstraps, sample_ready=True):
    """Locked GT에서 raw/aligned/refined Pose의 go/no-go를 판정합니다."""
    swap_rates = {row["model"]: row["swap_rate"] for row in swaps}
    bootstrap_map = {row["second"]: row for row in bootstraps}
    raw = select_summary_row(
        summary,
        model="raw",
        evaluation_set="locked",
        visibility_scope="V",
        joint_group="all",
    )
    decisions = []

    if raw is None or not finite_metric(raw["median_error"]):
        return decisions

    for model in ["raw", "aligned", "refined"]:
        overall = select_summary_row(
            summary,
            model=model,
            evaluation_set="locked",
            visibility_scope="V",
            joint_group="all",
        )
        anchor = select_summary_row(
            summary,
            model=model,
            evaluation_set="locked",
            visibility_scope="V",
            joint_group="shoulder_hip",
        )

        required = ["median_error", "p90_error", "pck_20", "catastrophic_30", "catastrophic_50", "coverage"]

        if overall is None or anchor is None:
            continue

        if not all(finite_metric(overall[value]) for value in required):
            continue

        if not finite_metric(anchor["median_error"]) or not finite_metric(anchor["coverage"]):
            continue

        swap_rate = swap_rates.get(model)
        normalized_error = overall["median_normalized_error"]
        absolute_families = True

        for group in sorted(set(POSE_JOINT_GROUPS.values())):
            family = select_summary_row(
                summary,
                model=model,
                evaluation_set="locked",
                visibility_scope="V",
                joint_group=group,
            )

            if family is None or not finite_metric(family["median_error"]) or not finite_metric(family["coverage"]):
                absolute_families = False
                break

            if family["coverage"] < 0.90:
                absolute_families = False
                break

        absolute = overall["median_error"] <= 15
        absolute |= finite_metric(normalized_error) and normalized_error <= 0.07
        absolute &= overall["pck_20"] >= 0.75
        absolute &= overall["p90_error"] <= 35
        absolute &= overall["catastrophic_50"] <= 0.10
        absolute &= overall["coverage"] >= 0.90
        absolute &= anchor["median_error"] <= 15
        absolute &= anchor["coverage"] >= 0.90
        absolute &= absolute_families
        absolute &= finite_metric(swap_rate) and swap_rate <= 0.02
        absolute &= sample_ready
        relative = None
        relative_improvement = None
        family_preserved = None

        if model != "raw":
            relative_improvement = (raw["median_error"] - overall["median_error"]) / max(
                raw["median_error"],
                1e-6,
            )
            pck_improvement = overall["pck_20"] - raw["pck_20"]
            catastrophic_reduction = (
                (raw["catastrophic_30"] - overall["catastrophic_30"]) / raw["catastrophic_30"]
                if raw["catastrophic_30"]
                else 0.0
            )
            family_preserved = True

            for group in sorted(set(POSE_JOINT_GROUPS.values())):
                baseline_group = select_summary_row(
                    summary,
                    model="raw",
                    evaluation_set="locked",
                    visibility_scope="V",
                    joint_group=group,
                )
                candidate_group = select_summary_row(
                    summary,
                    model=model,
                    evaluation_set="locked",
                    visibility_scope="V",
                    joint_group=group,
                )

                if baseline_group is None or candidate_group is None:
                    family_preserved = False
                    break

                if not all(
                    finite_metric(row[value])
                    for row in [baseline_group, candidate_group]
                    for value in ["median_error", "coverage"]
                ):
                    family_preserved = False
                    break

                if baseline_group["coverage"] < 0.90 or candidate_group["coverage"] < 0.90:
                    family_preserved = False
                    break

                allowance = max(2.0, baseline_group["median_error"] * 0.05)

                if candidate_group["median_error"] > baseline_group["median_error"] + allowance:
                    family_preserved = False

            bootstrap = bootstrap_map.get(model, {})
            confidence_preserved = finite_metric(bootstrap.get("ci_high"))

            if confidence_preserved:
                confidence_preserved = bootstrap["ci_high"] <= raw["median_error"] * 0.01

            raw_swap_rate = swap_rates.get("raw")
            swap_preserved = finite_metric(swap_rate) and finite_metric(raw_swap_rate)

            if swap_preserved:
                swap_preserved = swap_rate <= max(0.02, raw_swap_rate)

            relative = relative_improvement >= 0.05
            relative &= pck_improvement >= 0.02 or catastrophic_reduction >= 0.20
            relative &= family_preserved and confidence_preserved and swap_preserved
            relative &= sample_ready

        decisions.append(
            {
                "model": model,
                "absolute_pose_gate": bool(absolute),
                "relative_to_raw_gate": bool(relative) if relative is not None else None,
                "median_error": overall["median_error"],
                "relative_median_improvement": relative_improvement,
                "pck_20": overall["pck_20"],
                "p90_error": overall["p90_error"],
                "catastrophic_50": overall["catastrophic_50"],
                "swap_rate": swap_rate,
                "joint_families_preserved": family_preserved,
                "minimum_sample_reached": bool(sample_ready),
            }
        )

    return decisions


def roi_decisions(errors, summary, quality, sample_ready=True):
    """Locked GT에서 ROI geometry와 intensity source의 go/no-go를 판정합니다."""
    overall = select_summary_row(
        summary,
        candidate="selected",
        evaluation_set="locked",
        visibility_scope="V+I",
        zone_group="all",
        source="all",
    )
    groups = []

    for group in sorted(set(ROI_ZONE_GROUPS.values())):
        current = select_summary_row(
            summary,
            candidate="selected",
            evaluation_set="locked",
            visibility_scope="V+I",
            zone_group=group,
            source="all",
        )

        if current is not None:
            groups.append(current)

    opposite = errors[
        (errors["candidate"] == "selected")
        & (errors["evaluation_set"] == "locked")
        & (errors["zone_id"].isin([5, 17]))
        & (errors["visibility"].isin(["N", "O"]))
    ]
    opposite_false_activation = int(opposite["predicted_visible"].sum())
    intensity = errors[
        (errors["candidate"] == "selected")
        & (errors["source"] == "intensity")
        & (errors["evaluation_set"] == "locked")
        & (errors["visibility"].isin(["V", "I"]))
    ]
    keys = ["scan_id", "view_index", "zone_id"]
    oriented = errors[(errors["candidate"] == "oriented_counterfactual") & (errors["evaluation_set"] == "locked")]
    comparison = intensity.merge(oriented[keys + ["iou", "coverage"]], on=keys, suffixes=("_intensity", "_oriented"))
    comparison = comparison.dropna(subset=["iou_intensity", "iou_oriented"])
    intensity_delta = None
    intensity_coverage_delta = None
    intensity_gate = None

    if len(comparison):
        intensity_delta = float(comparison["iou_intensity"].median() - comparison["iou_oriented"].median())
        intensity_coverage_delta = float(
            comparison["coverage_intensity"].median() - comparison["coverage_oriented"].median()
        )
        intensity_catastrophic = float((comparison["iou_intensity"] < 0.20).mean())
        oriented_catastrophic = float((comparison["iou_oriented"] < 0.20).mean())
        intensity_gate = intensity_delta >= 0.03
        intensity_gate &= intensity_coverage_delta >= -0.02
        intensity_gate &= intensity_catastrophic <= oriented_catastrophic
        intensity_gate &= sample_ready

    geometry_gate = False

    if overall is not None and finite_metric(overall["median_iou"]):
        geometry_gate = overall["median_iou"] >= 0.55
        geometry_gate &= overall["median_coverage"] >= 0.85
        geometry_gate &= overall["catastrophic_020"] <= 0.10
        geometry_gate &= all(finite_metric(row["median_iou"]) and row["median_iou"] >= 0.50 for row in groups)
        geometry_gate &= opposite_false_activation == 0
        geometry_gate &= sample_ready

    quality_gate = finite_metric(quality["auroc_iou_050"]) and quality["auroc_iou_050"] >= 0.70
    quality_gate &= finite_metric(quality["spearman"]) and quality["spearman"] >= 0.30
    quality_gate &= sample_ready

    return {
        "phase26_geometry_gate": bool(geometry_gate),
        "intensity_profile_gate": bool(intensity_gate) if intensity_gate is not None else None,
        "quality_weighting_gate": bool(quality_gate),
        "minimum_sample_reached": bool(sample_ready),
        "opposite_torso_false_activations": opposite_false_activation,
        "intensity_vs_oriented_median_iou_delta": intensity_delta,
        "intensity_vs_oriented_median_coverage_delta": intensity_coverage_delta,
        "locked_overall": overall,
        "locked_zone_groups": groups,
    }


def evaluate_ground_truth(
    artifact_file,
    ground_truth_directory,
    output_directory,
    bootstrap_iterations=2000,
    random_seed=42,
    allow_artifact_mismatch=False,
):
    """사람 GT로 Pose 후보와 ROI geometry를 분리 평가합니다."""
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be one or greater")

    ground_truth_directory = Path(ground_truth_directory)
    manifest = load_annotation_manifest(
        ground_truth_directory / "annotation_manifest.csv",
        ground_truth_directory,
    )
    pose_ground_truth = load_pose_ground_truth(ground_truth_directory / "pose_ground_truth.csv")
    roi_ground_truth = load_roi_ground_truth(ground_truth_directory / "roi_ground_truth.csv")
    validate_ground_truth_coverage(manifest, pose_ground_truth, roi_ground_truth)
    provenance_file = ground_truth_directory / "provenance.json"

    if not provenance_file.exists():
        raise ValueError("GT provenance.json is missing")

    with open(provenance_file, encoding="utf-8") as file:
        provenance = json.load(file)

    validate_ground_truth_provenance(manifest, provenance)

    if manifest["scan_id"].nunique() != provenance.get("sample_count"):
        raise ValueError("Annotation manifest scan count differs from GT provenance")

    roi_scan_count = manifest.loc[manifest["annotate_roi"], "scan_id"].nunique()

    if roi_scan_count != provenance.get("roi_scan_count"):
        raise ValueError("Annotation manifest ROI scan count differs from GT provenance")

    if provenance.get("coordinate_system") != {"height": HEIGHT, "width": WIDTH, "origin": "top_left"}:
        raise ValueError("GT provenance coordinate system differs from the native artifact contract")

    with np.load(artifact_file, allow_pickle=False) as artifact:
        errors = validate_aggregate(artifact)

        if errors:
            raise ValueError("Invalid artifact: " + "; ".join(errors))

        artifact_matches = provenance.get("artifact_sha256") == file_sha256(artifact_file)

        if not artifact_matches and not allow_artifact_mismatch:
            raise ValueError("Artifact differs from GT provenance. Use --allow-artifact-mismatch intentionally")

        pose_errors = pose_error_table(artifact, pose_ground_truth)
        roi_errors = roi_error_table(artifact, roi_ground_truth)

    pose_summary = pose_summary_table(pose_errors)
    pose_actions, pose_action_summary_data = pose_action_summary(pose_errors)
    roi_summary = roi_summary_table(roi_errors)
    swaps = pose_swap_rates(pose_errors)
    sample_status = evaluation_sample_status(pose_ground_truth, roi_ground_truth)
    pose_bootstrap = [
        paired_pose_bootstrap(
            pose_errors,
            "raw",
            model,
            bootstrap_iterations,
            random_seed + index,
        )
        for index, model in enumerate(["aligned", "refined"])
    ]
    roi_bootstrap = [
        roi_scan_bootstrap(
            roi_errors,
            candidate,
            bootstrap_iterations,
            random_seed + index,
        )
        for index, candidate in enumerate(["selected", "oriented_counterfactual", "relative_counterfactual"])
    ]
    quality = roi_quality_calibration(roi_errors)
    decisions = {
        "artifact_matches_annotation_provenance": artifact_matches,
        "evaluation_sample": sample_status,
        "pose": pose_decisions(pose_summary, swaps, pose_bootstrap, sample_status["pose_ready"]),
        "pose_actions": pose_action_summary_data.to_dict(orient="records"),
        "roi": roi_decisions(roi_errors, roi_summary, quality, sample_status["roi_ready"]),
        "pose_bootstrap": pose_bootstrap,
        "roi_bootstrap": roi_bootstrap,
        "pose_swap_rates": swaps,
        "roi_quality_calibration": quality,
    }

    output_directory = Path(output_directory)
    write_csv(output_directory / "pose_errors.csv", pose_errors)
    write_csv(output_directory / "pose_summary.csv", pose_summary)
    write_csv(output_directory / "pose_actions.csv", pose_actions)
    write_csv(output_directory / "pose_action_summary.csv", pose_action_summary_data)
    write_csv(output_directory / "roi_errors.csv", roi_errors)
    write_csv(output_directory / "roi_summary.csv", roi_summary)
    write_json(output_directory / "decisions.json", serializable_report(decisions))
    print(json.dumps(serializable_report(decisions), ensure_ascii=False, indent=2))
    return decisions


def argument_parser():
    """Pose/ROI 생성과 검사를 위한 명령어를 정의합니다."""
    parser = argparse.ArgumentParser(description="Generate native TSA Pose and ROI artifacts")
    commands = parser.add_subparsers(dest="command")

    preview = commands.add_parser("preview")
    preview.add_argument("--dataset", default="data/splits/dataset.csv")
    preview.add_argument("--data-directory", default="data")
    preview.add_argument("--model", default="models/yolov8x-pose.pt")
    preview.add_argument("--results", default=f"results/roi/{ARTIFACT_VERSION}/preview")
    preview.add_argument("--sample-count", type=int, default=10)

    export = commands.add_parser("export")
    export.add_argument("--dataset", default="data/splits/dataset.csv")
    export.add_argument("--data-directory", default="data")
    export.add_argument("--model", default="models/yolov8x-pose.pt")
    export.add_argument("--output", default=f"data/roi/{ARTIFACT_VERSION}")
    export.add_argument("--shard-index", type=int, default=0)
    export.add_argument("--shard-count", type=int, default=1)
    export.add_argument("--limit", type=int)
    export.add_argument("--force", action="store_true")

    collect = commands.add_parser("collect")
    collect.add_argument("--dataset", default="data/splits/dataset.csv")
    collect.add_argument("--source", action="append", required=True)
    collect.add_argument("--output", default=f"data/roi/{ARTIFACT_VERSION}")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--dataset", default="data/splits/dataset.csv")
    finalize.add_argument("--output", default=f"data/roi/{ARTIFACT_VERSION}")
    finalize.add_argument("--allow-partial", action="store_true")

    audit = commands.add_parser("audit")
    audit.add_argument("--dataset", default="data/splits/dataset.csv")
    audit.add_argument("--artifact", default=f"data/roi/{ARTIFACT_VERSION}/{ARTIFACT_VERSION}.npz")
    audit.add_argument("--results", default=f"results/roi/{ARTIFACT_VERSION}")
    audit.add_argument("--allow-partial", action="store_true")

    ground_truth = commands.add_parser("prepare-gt")
    ground_truth.add_argument("--dataset", default="data/splits/dataset.csv")
    ground_truth.add_argument("--data-directory", default="data")
    ground_truth.add_argument("--artifact", default=f"data/roi/{ARTIFACT_VERSION}/{ARTIFACT_VERSION}.npz")
    ground_truth.add_argument("--output", default=f"data/roi/{ARTIFACT_VERSION}/ground_truth")
    ground_truth.add_argument("--sample-count", type=int, default=40)
    ground_truth.add_argument("--roi-scan-count", type=int, default=20)
    ground_truth.add_argument("--random-seed", type=int, default=42)
    ground_truth.add_argument("--force", action="store_true")

    evaluate = commands.add_parser("evaluate-gt")
    evaluate.add_argument("--artifact", default=f"data/roi/{ARTIFACT_VERSION}/{ARTIFACT_VERSION}.npz")
    evaluate.add_argument("--ground-truth", default=f"data/roi/{ARTIFACT_VERSION}/ground_truth")
    evaluate.add_argument("--results", default=f"results/roi/{ARTIFACT_VERSION}/ground_truth")
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate.add_argument("--random-seed", type=int, default=42)
    evaluate.add_argument("--allow-artifact-mismatch", action="store_true")

    pose_training = commands.add_parser("prepare-pose-training")
    pose_training.add_argument("--dataset", default="data/splits/dataset.csv")
    pose_training.add_argument("--artifact", default=f"data/roi/{ARTIFACT_VERSION}/{ARTIFACT_VERSION}.npz")
    pose_training.add_argument("--ground-truth", default=f"data/roi/{ARTIFACT_VERSION}/ground_truth")
    pose_training.add_argument("--output", default=f"data/roi/{ARTIFACT_VERSION}/pose_training")
    pose_training.add_argument("--random-seed", type=int, default=42)
    pose_training.add_argument("--force", action="store_true")

    return parser


def main():
    parser = argument_parser()
    arguments = parser.parse_args()

    if arguments.command is None:
        parser.print_help()
        return

    if arguments.command == "preview":
        preview_samples(
            arguments.dataset,
            arguments.data_directory,
            arguments.model,
            arguments.results,
            arguments.sample_count,
        )
    elif arguments.command == "export":
        export_artifacts(
            arguments.dataset,
            arguments.data_directory,
            arguments.model,
            arguments.output,
            arguments.shard_index,
            arguments.shard_count,
            arguments.limit,
            arguments.force,
        )
    elif arguments.command == "collect":
        collect_checkpoints(arguments.dataset, arguments.source, arguments.output)
    elif arguments.command == "finalize":
        finalize_artifacts(arguments.dataset, arguments.output, arguments.allow_partial)
    elif arguments.command == "audit":
        allow_partial = arguments.allow_partial or arguments.artifact.endswith(".partial.npz")
        audit = audit_artifacts(arguments.dataset, arguments.artifact, arguments.results, allow_partial)

        if not audit["all_checks_passed"]:
            raise SystemExit(1)
    elif arguments.command == "prepare-gt":
        prepare_ground_truth(
            arguments.dataset,
            arguments.data_directory,
            arguments.artifact,
            arguments.output,
            arguments.sample_count,
            arguments.roi_scan_count,
            arguments.random_seed,
            arguments.force,
        )
    elif arguments.command == "evaluate-gt":
        evaluate_ground_truth(
            arguments.artifact,
            arguments.ground_truth,
            arguments.results,
            arguments.bootstrap_iterations,
            arguments.random_seed,
            arguments.allow_artifact_mismatch,
        )
    elif arguments.command == "prepare-pose-training":
        prepare_pose_training_data(
            arguments.dataset,
            arguments.artifact,
            arguments.ground_truth,
            arguments.output,
            arguments.random_seed,
            arguments.force,
        )


if __name__ == "__main__":
    main()
