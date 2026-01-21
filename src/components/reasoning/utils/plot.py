import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import random

SKELETON = [
    (0, 13),  # Nose -> Neck
    (1, 2),  # LShoulder -> RShoulder
    (1, 3),
    (3, 5),  # LShoulder -> LElbow -> LWrist
    (2, 4),
    (4, 6),  # RShoulder -> RElbow -> RWrist
    (13, 7),
    (7, 9),
    (9, 11),  # Neck -> LHip -> LKnee -> LAnkle
    (13, 8),
    (8, 10),
    (10, 12),  # Neck -> RHip -> RKnee -> RAnkle
]

LINE_COLORS = [
    (0, 215, 255),
    (0, 255, 204),
    (0, 134, 255),
    (0, 255, 50),
    (77, 255, 222),
    (77, 196, 255),
    (77, 135, 255),
    (191, 255, 77),
    (77, 255, 77),
    (77, 222, 255),
    (255, 156, 127),
    (0, 127, 255),
    (255, 127, 77),
    (0, 77, 255),
    (255, 77, 36),
]

POINT_COLORS = [
    (0, 255, 255),
    (0, 191, 255),
    (0, 255, 102),
    (0, 77, 255),
    (0, 255, 0),  # Nose, LEye, REye, LEar, REar
    (77, 255, 255),
    (77, 255, 204),
    (77, 204, 255),
    (191, 255, 77),
    (77, 191, 255),
    (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
    (204, 77, 255),
    (77, 255, 204),
    (191, 77, 255),
    (77, 255, 191),
    (127, 77, 255),
    (77, 255, 127),
    (0, 255, 255),
]

CLASS_COLORS = {
    "body": (0, 255, 255),   # cyan
    "head": (255, 255, 0),   # yellow
    "person": (0, 255, 0),
    "CRITICAL_FALL": (0, 0, 255),  # red
    "Fall Down": (0, 0, 255),
    "Lying Down": (0, 0, 255),
}

def get_distinct_colors(n: int = 300) -> List[Tuple[int, int, int]]:
    forbidden = [(0,10),(170,179),(20,35),(35,85)]
    def is_forbidden(h): return any(s <= h <= e for s,e in forbidden)
    colors = []
    for i in range(n):
        for _ in range(100):
            h = random.randint(0, 179)
            if not is_forbidden(h):
                s = random.randint(180, 255)
                v = random.randint(180, 255)
                bgr = cv2.cvtColor(np.array([[[h,s,v]]], np.uint8), cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, bgr)))
                break
        else:
            colors.append((255, 100, 100))  # fallback
    return colors

TRACKING_COLORS = get_distinct_colors(500)

def draw_smart_bbox(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    alpha: float = 0.7
):
    overlay = img.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    text_thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    tw += 14

    label_x = x1
    label_y = y1 - th - 12
    place_below = False
    if label_y < 10:
        place_below = True
        label_y = y1 + th + 12

    label_x2 = label_x + tw
    if label_x2 > img.shape[1]:
        label_x = img.shape[1] - tw - 5
    label_x = max(5, label_x)

    if place_below:
        bg_top = y1
        bg_bottom = y1 + th + 12
    else:
        bg_top = y1 - th - 16
        bg_bottom = y1

    cv2.rectangle(overlay, (label_x, bg_top), (label_x + tw, bg_bottom), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    text_y = bg_top + th + 5 if place_below else y1 - 8
    cv2.putText(img, label, (label_x + 7, text_y),
                font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_pose(
    img: np.ndarray,
    keypoints: List[dict],
    threshold: float = 0.1,
):
    if not keypoints:
        return img

    if len(keypoints) < 14:
        keypoints += [[0.0, 0.0, 0.0]] * (14 - len(keypoints))

    left_shoulder = keypoints[1]
    right_shoulder = keypoints[2]

    if left_shoulder[2] > threshold and right_shoulder[2] > threshold:
        neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
        neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
        neck_conf = min(left_shoulder[2], right_shoulder[2])
        keypoints[13] = [neck_x, neck_y, neck_conf]

    for i, (p1, p2) in enumerate(SKELETON):
        if p1 >= len(keypoints) or p2 >= len(keypoints):
            continue
        if keypoints[p1][2] <= threshold or keypoints[p2][2] <= threshold:
            continue

        pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
        pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))

        thickness = int(2 * (keypoints[p1][2] + keypoints[p2][2]) + 1)

        line_color = LINE_COLORS[i % len(LINE_COLORS)]
        point_color_1 = POINT_COLORS[p1 % len(POINT_COLORS)]
        point_color_2 = POINT_COLORS[p2 % len(POINT_COLORS)]

        cv2.line(img, pt1, pt2, line_color, thickness)
        cv2.circle(img, pt1, 3, point_color_1, -1)
        cv2.circle(img, pt2, 3, point_color_2, -1)

    return img

def draw_detection(result, img):
    detections = result.get("detection", [])
    for det in detections:
        label = det.get("label", "unknown")
        conf = det.get("confidence", 0.0)
        box = det.get("box")
        if not box: continue
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(label, (100, 100, 255))
        text = f"{label} {conf:.2f}"
        draw_smart_bbox(img, (x1, y1, x2, y2), text, color, thickness=2)

def draw_action(result, modes, img):
    has_critical_fall = False
    tracks = result.get("track", [])
    action_events = result.get("action_events", [])

    action_map = {e["track_id"]: e for e in action_events if e.get("track_id")}

    for obj in tracks:
        box = obj.get("box")
        if not box: continue
        x1, y1, x2, y2 = box
        tid = obj.get("track_id", "???")

        color = TRACKING_COLORS[abs(hash(str(tid))) % len(TRACKING_COLORS)]
        parts = [f"ID:{tid}"]

        # === ACTION ===
        if "action" in modes:
            act = action_map.get(tid)
            if act:
                label = act.get("label", "tracking")
                if label.lower() in ["fall down", "lying down"]:
                    color = CLASS_COLORS["CRITICAL_FALL"]
                    parts[-1] = "FALL DOWN!"
                    has_critical_fall = True
                else:
                    label = "Tracking"

        final_label = " | ".join(parts)
        draw_smart_bbox(img, (x1, y1, x2, y2), final_label, color, thickness=3)

        # === POSE ===
        if "pose" in modes:
            kps = obj.get("keypoints")
            if kps and len(kps) >= 13:
                img = draw_pose(img, kps, threshold=0.01)
    
    return img, has_critical_fall

def draw_counting(result, img, top_offset=0):
    count_info = result.get("count", {})
    current_inside = count_info.get("current_inside", {})
    polygon_data = count_info.get("polygon", {})

    h, w = img.shape[:2]
    scale = max(w / 1920, h / 1080, 0.7)

    if polygon_data:
        coordinates = polygon_data.get("coordinates", [])
        for poly_coords in coordinates:
            pts = np.array(poly_coords, np.int32).reshape((-1, 1, 2))

            overlay = img.copy()
            cv2.polylines(overlay, [pts], True, (0, 255, 0), thickness=int(4 * scale))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
            cv2.polylines(img, [pts], True, (0, 255, 0), thickness=int(2 * scale))

    total_inside = sum(v for v in current_inside.values()) if current_inside else 0

    if total_inside == 0:
        return

    lines = []
    main_text = f"INSIDE ZONE: {total_inside}"
    lines.append(main_text)

    if current_inside:
        detail = " • ".join([f"{k.capitalize()}: {v}" for k, v in current_inside.items()])
        lines.append(detail)

    count_in = count_info.get("count_in", {})
    count_out = count_info.get("count_out", {})
    in_total = sum(count_in.values()) if isinstance(count_in, dict) else 0
    out_total = sum(count_out.values()) if isinstance(count_out, dict) else 0

    if in_total or out_total:
        lines.append(f"IN → {in_total}    OUT ← {out_total}")

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.1 * scale
    thickness = max(2, int(2.5 * scale))
    line_height = int(55 * scale)
    padding_x = int(25 * scale)
    padding_y = int(20 * scale)
    corner_radius = int(15 * scale)

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(size[0] for size in text_sizes) + 2 * padding_x
    total_height = len(lines) * line_height + 2 * padding_y

    box_x1 = int(20 * scale)
    box_y1 = int(top_offset + 20 * scale)
    box_x2 = box_x1 + max_width
    box_y2 = box_y1 + total_height

    overlay = img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (200, 200, 200), thickness=int(3 * scale))

    radius = corner_radius
    cv2.ellipse(img, (box_x1 + radius, box_y1 + radius), (radius, radius), 0, 180, 270, (200, 200, 200), thickness=int(3 * scale))
    cv2.ellipse(img, (box_x2 - radius, box_y1 + radius), (radius, radius), 0, 270, 360, (200, 200, 200), thickness=int(3 * scale))
    cv2.ellipse(img, (box_x1 + radius, box_y2 - radius), (radius, radius), 0, 90, 180, (200, 200, 200), thickness=int(3 * scale))
    cv2.ellipse(img, (box_x2 - radius, box_y2 - radius), (radius, radius), 0, 0, 90, (200, 200, 200), thickness=int(3 * scale))


    y = box_y1 + padding_y + int(35 * scale)
    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        if i == 2:
            color = (180, 255, 180)
        cv2.putText(img, line, (box_x1 + padding_x, y),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
    
    return box_y2

def draw_critical_fall(img, scale, t):
    h, w = img.shape[:2]
    overlay = img.copy()

    alpha = 0.4 + 0.3 * np.sin(t * 10)
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    text = "CRITICAL FALL!"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.8 * scale
    thickness = max(3, int(5 * scale))

    flash = int(t * 10) % 2 == 0
    color = (0, 0, 255) if flash else (0, 255, 255)

    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y = int(90 * scale)

    cv2.putText(img, text, ((w - tw) // 2, y),
                font, font_scale, color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, ((w - tw) // 2, y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_fire(img, scale, t):
    h, w = img.shape[:2]
    overlay = img.copy()

    alpha = 0.4 + 0.3 * np.sin(t * 12)
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    border_thickness = 15 + int(10 * np.sin(t * 15))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1),
                  (0, 0, 255), border_thickness)

    text = "FIRE DETECTED!!!"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2.8 * scale
    thickness = max(4, int(8 * scale))

    flash = int(t * 10) % 2 == 0
    color = (0, 0, 255) if flash else (0, 255, 255)

    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y = int(130 * scale)

    cv2.putText(img, text, ((w - tw) // 2, y),
                font, font_scale, color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, ((w - tw) // 2, y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_fire_warning(img, scale):
    text = "FIRE WARNING"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.6 * scale
    thickness = max(3, int(5 * scale))

    x = int(20 * scale)
    y = int(60 * scale)

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - th - 10),
        (x + tw + 10, y + 10),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, text, (x, y),
                font, font_scale, (0, 255, 255),
                thickness, cv2.LINE_AA)

    return y + 20

def draw_alerts(img, scale, t, has_critical_fall, fire_state):
    top_offset = 0

    if has_critical_fall:
        draw_critical_fall(img, scale, t)

    if fire_state == "fire":
        draw_fire(img, scale, t)

    elif fire_state == "warning":
        top_offset = draw_fire_warning(img, scale)

    return top_offset

def draw_faces(result, img):
    face_events = result.get("face_event", [])
    face_detection = result.get("face_detection", [])
    face_color_map = {}
    for event in face_events:
        pid = event.get("personal_id", "Unknown")
        if pid not in face_color_map:
            if pid == "Unknown":
                face_color_map[pid] = (100, 100, 255)
            else:
                idx = abs(hash(pid)) % len(TRACKING_COLORS)
                face_color_map[pid] = TRACKING_COLORS[idx]


    for i, event in enumerate(face_events):
        if i >= len(face_detection):
            break
        det = face_detection[i]

        if det.get("label") != "face":
            continue
        box = det.get("box")
        if not box:
            continue
        
        x1, y1, x2, y2 = box
        pid = event.get("personal_id", "Unknown")
        ptype = event.get("person_type", "unknown").capitalize()
        conf = event.get("confidence", 0.0)

        label = f"{pid} ({ptype})"
        # color = (0, 255, 255) if ptype != "Unknown" else (100, 100, 255)
        color = face_color_map.get(pid, (100, 100, 255))

        draw_smart_bbox(img, (x1, y1, x2, y2), label, color, thickness=2)

def visualize(
    frame: np.ndarray,
    result: Dict[str, Any],
    modes: List[str] = ("tracking", "pose", "action", "face", "counting", "fire"),
    show_critical_alert: bool = True,
    inv_scale: bool = True,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    base_w = 1920
    base_h = 1080
    scale = max(w / base_w, h / base_h, 0.5)
    current_time = time.time()

    fire_data = result.get("fire", {})
    global_state = fire_data.get("state", "normal")  # "fire", "warning", "normal"
    fire_confidence = fire_data.get("confidence", 0.0)

    modes = set(modes)

    # === DETECTION ===
    if "detection" in modes and len(modes) == 1:
        draw_detection(result, img)

    has_critical_fall = False
    # === TRACKING + ACTION + FACE + POSE ===
    if any(m in modes for m in ["tracking", "pose", "action", "face"]):
        img, has_critical_fall = draw_action(result, modes, img)


    # === FACE VISUALIZATION ===
    if "face" in modes:
        draw_faces(result, img)

    # === CRITICAL ALERT OVERLAY ===
    top_offset = 0
    if show_critical_alert:
        top_offset = draw_alerts(
            img,
            scale,
            current_time,
            has_critical_fall,
            global_state
        )
    
    # === COUNTING VISUALIZATION ===
    if "counting" in modes or len(modes) == 0:
        draw_counting(result, img, top_offset)

    if not inv_scale:
        return img

    ori_scale = 1 / result.get("scale_factor", 1)
    new_w = int(img.shape[1] * ori_scale)
    new_h = int(img.shape[0] * ori_scale)

    return cv2.resize(img, (new_w, new_h))