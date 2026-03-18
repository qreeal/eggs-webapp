import os
import time
from pathlib import Path
import base64

import cv2
import numpy as np
import math

from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO, FastSAM
import pytesseract

# Если нужно, укажи путь к системному tesseract
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


# ==========================
# 1. ЛОГРЕГ (BGR-версия)
# ==========================
FEATURE_COLS = [
    "mean_L", "std_L", "mean_a", "std_a", "mean_b", "std_b",
    "mean_H", "std_H", "mean_S", "std_S", "mean_V", "std_V",
    "edge_density", "ratio", "area", "h", "w",
]

CLASS_NAMES_LR = ["egg", "red_tomato", "yellow_tomato"]

W_lr = np.array([
    [ 2.112813,  0.976787,  3.654014, -0.659353,  4.609894, -0.831808,
      5.687173,  1.924252, -6.119059,  1.068493, -7.016335,  3.742521,
      0.072778,  0.031098,  0.024102,  1.443688,  4.368437],   # egg
    [-3.631302,  3.084834, -1.083466,  1.243061, -4.792377,  0.878610,
      7.676166,  6.751149,  1.066931,  1.923943,  0.428029,  1.162680,
     -0.077870, -0.000433,  0.198493,  0.031844, -3.743525],   # red_tomato
    [ 1.518489, -4.061622, -2.570548, -0.583708,  0.182483, -0.046802,
    -13.363339, -8.675401,  5.052128, -2.992435,  6.588306, -4.905200,
      0.005092, -0.030664, -0.222595, -1.475532, -0.624911],   # yellow_tomato
], dtype=np.float64)

B_lr = np.array([0.098330, -0.057727, -0.040604], dtype=np.float64)


def compute_features_rich_bgr(crop_bgr):
    h, w = crop_bgr.shape[:2]
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    center = crop_bgr[cy1:cy2, cx1:cx2]
    if center.size == 0:
        center = crop_bgr

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)

    lab = cv2.cvtColor(center, cv2.COLOR_BGR2LAB)
    Lc, ac, bc = cv2.split(lab)

    V_norm = cv2.normalize(Vc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(V_norm, 50, 150)

    return dict(
        mean_L=float(Lc.mean()), std_L=float(Lc.std()),
        mean_a=float(ac.mean()), std_a=float(ac.std()),
        mean_b=float(bc.mean()), std_b=float(bc.std()),
        mean_H=float(Hc.mean()), std_H=float(Hc.std()),
        mean_S=float(Sc.mean()), std_S=float(Sc.std()),
        mean_V=float(Vc.mean()), std_V=float(Vc.std()),
        edge_density=float((edges > 0).mean()),
        ratio=min(h, w) / max(h, w),
        area=float(h * w),
        h=h, w=w,
    )


def classify_logreg_3class(feats_dict):
    x = np.array([feats_dict[name] for name in FEATURE_COLS], dtype=np.float64)
    z = W_lr @ x + B_lr
    return CLASS_NAMES_LR[int(np.argmax(z))]


# ==========================
# 2. МАСКИ / FASTSAM / ТОМАТЫ
# ==========================

def mask_s_otsu(crop_rgb):
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    S = crop_hsv[:, :, 1]
    _, m = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def mask_dizenzo_otsu(crop_rgb):
    blurred = cv2.GaussianBlur(crop_rgb, (5, 5), 0)
    B, G, R = cv2.split(blurred.astype(np.float32))

    Rx = cv2.Sobel(R, cv2.CV_32F, 1, 0, ksize=3)
    Ry = cv2.Sobel(R, cv2.CV_32F, 0, 1, ksize=3)
    Gx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
    Bx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)
    By = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)

    g_xx = Rx**2 + Gx**2 + Bx**2
    g_yy = Ry**2 + Gy**2 + By**2
    g_xy = Rx*Ry + Gx*Gy + Bx*By

    F = np.sqrt(
        0.5 * ((g_xx + g_yy) + np.sqrt((g_xx - g_yy)**2 + 4 * g_xy**2 + 1e-12))
    )
    F_norm = cv2.normalize(F, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, m = cv2.threshold(F_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def mask_color_red_wide(crop_rgb):
    """
    Широкая маска красно‑оранжевых оттенков под помидоры.
    Вход: crop_rgb в формате RGB.
    Выход: бинарная маска uint8 (0 или 255).
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    lower_r1 = np.array([0,   40, 30], dtype=np.uint8)
    upper_r1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_r2 = np.array([170, 40, 30], dtype=np.uint8)
    upper_r2 = np.array([179, 255, 255], dtype=np.uint8)

    lower_or = np.array([10,  20, 20], dtype=np.uint8)
    upper_or = np.array([25, 255, 255], dtype=np.uint8)

    mask_r1 = cv2.inRange(hsv, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsv, lower_r2, upper_r2)
    mask_or = cv2.inRange(hsv, lower_or, upper_or)

    m = cv2.bitwise_or(mask_r1, mask_r2)
    m = cv2.bitwise_or(m, mask_or)

    return m


def adaptive_red_mask(crop_rgb, seed_mask, dH=15, dS=40, dV=40):
    if seed_mask is None or seed_mask.max() == 0:
        return np.zeros(crop_rgb.shape[:2], dtype=np.uint8)

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    mask_bool = seed_mask > 0
    H = hsv[:, :, 0][mask_bool]
    S = hsv[:, :, 1][mask_bool]
    V = hsv[:, :, 2][mask_bool]

    if H.size == 0:
        return np.zeros(crop_rgb.shape[:2], dtype=np.uint8)

    H_med = np.median(H)
    S_med = np.median(S)
    V_med = np.median(V)

    def hue_mask(h):
        diff = np.abs(h.astype(np.int16) - int(H_med))
        diff = np.minimum(diff, 180 - diff)
        return diff <= dH

    H_mask = hue_mask(hsv[:, :, 0])
    S_mask = np.abs(hsv[:, :, 1].astype(np.int16) - int(S_med)) <= dS
    V_mask = np.abs(hsv[:, :, 2].astype(np.int16) - int(V_med)) <= dV

    m = (H_mask & S_mask & V_mask).astype(np.uint8) * 255
    return m


def score_mask_tomato(mask):
    h, w = mask.shape[:2]
    if mask.max() == 0:
        return -1e9

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return -1e9

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    area = stats[idx, cv2.CC_STAT_AREA]
    x, y, ww, hh = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP], \
                   stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[idx]

    cx0, cy0 = w / 2.0, h / 2.0
    center_dist = np.hypot(cx - cx0, cy - cy0) / max(w, h)

    r = max(ww, hh) / 2.0
    if r <= 0:
        return -1e9
    compact = area / (np.pi * r * r)

    area_norm = area / float(w * h)
    score = (
        2.0 * area_norm +
        2.0 * compact -
        3.0 * center_dist
    )
    return score


def fill_all_inner_holes(mask):
    m = mask.copy().astype(np.uint8)
    if m.max() == 0:
        return m

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)

    comp = np.zeros_like(m, dtype=np.uint8)
    comp[labels == idx] = 255

    inv = cv2.bitwise_not(comp)

    h, w = inv.shape[:2]
    inv_flood = inv.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    for x in range(w):
        if inv_flood[0, x] == 255:
            cv2.floodFill(inv_flood, mask_ff, (x, 0), 128)
        if inv_flood[h-1, x] == 255:
            cv2.floodFill(inv_flood, mask_ff, (x, h-1), 128)
    for y in range(h):
        if inv_flood[y, 0] == 255:
            cv2.floodFill(inv_flood, mask_ff, (0, y), 128)
        if inv_flood[y, w-1] == 255:
            cv2.floodFill(inv_flood, mask_ff, (w-1, y), 128)

    inner_holes = (inv_flood == 255).astype(np.uint8) * 255
    filled = cv2.bitwise_or(comp, inner_holes)
    return filled


def choose_best_mask_by_roundness(cand_dict, min_area=50):
    best_name = None
    best_mask = None
    best_score = -1.0
    scores = {}

    for name, m in cand_dict.items():
        m_u8 = m.astype(np.uint8)
        if m_u8.max() == 0:
            scores[name] = (0.0, 0.0, 0.0)
            continue

        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            scores[name] = (0.0, 0.0, 0.0)
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)

        if area < min_area or perim <= 0:
            scores[name] = (0.0, area, perim)
            continue

        circularity = 4.0 * math.pi * area / (perim * perim)
        scores[name] = (circularity, area, perim)

        score = circularity
        if score > best_score:
            best_score = score
            best_name = name
            best_mask = m_u8

    if best_mask is None:
        for name, m in cand_dict.items():
            area = (m > 0).sum()
            if area > best_score:
                best_score = area
                best_name = name
                best_mask = m

    return best_mask, best_name, scores


# ==========================
# 3. FASTSAM
# ==========================
# Путь к FastSAM весам поправь под себя
FASTSAM_WEIGHTS = "model/FastSAM-x.pt"
fastsam_model = FastSAM(FASTSAM_WEIGHTS)


def mask_fastsam(crop_rgb, conf=0.4, iou=0.9, imgsz=512):
    h, w = crop_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8)

    results = fastsam_model(
        crop_rgb,
        device="cpu",
        retina_masks=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    res = results[0]
    if res.masks is None or res.masks.data is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = res.masks.data.cpu().numpy()

    if masks.ndim != 3 or masks.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(areas))
    best = masks[best_idx]

    best = (best > 0.5).astype(np.uint8) * 255

    if best.shape != (h, w):
        best = cv2.resize(best, (w, h), interpolation=cv2.INTER_NEAREST)

    return best


def best_tomato_mask(crop_rgb):
    cand = {
        "S_otsu": mask_s_otsu(crop_rgb),
        # "dizenzo_otsu": mask_dizenzo_otsu(crop_rgb),
        "red_color": mask_color_red_wide(crop_rgb),
        "fastsam": mask_fastsam(crop_rgb),
    }
    best_mask, best_name, scores = choose_best_mask_by_roundness(cand)
    return best_mask, best_name, cand


# ==========================
# 4. СГЛАЖИВАНИЕ МАСОК / NMS
# ==========================

def smooth_contour_round(mask, smooth_factor=0.4, interp_points=400, inflate_iters=1):
    m = mask.copy().astype(np.uint8)
    if m.max() == 0:
        return m

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)

    comp = np.zeros_like(m)
    comp[labels == idx] = 255

    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return m

    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt[:, 0, :]

    cnt_closed = np.vstack([cnt, cnt[0]])

    diffs = np.diff(cnt_closed, axis=0)
    seg_len = np.sqrt((diffs**2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(seg_len)])
    if t[-1] == 0:
        return comp
    t /= t[-1]

    t_new = np.linspace(0, 1, interp_points)

    x = cnt_closed[:, 0]
    y = cnt_closed[:, 1]
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    pts = np.stack([x_new, y_new], axis=1)

    k = max(5, int(interp_points * smooth_factor) | 1)
    pad = k // 2
    pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])

    kernel = np.ones(k) / k
    xs = np.convolve(pts_pad[:, 0], kernel, mode="valid")
    ys = np.convolve(pts_pad[:, 1], kernel, mode="valid")

    smooth_pts = np.stack([xs, ys], axis=1).astype(np.int32)

    smooth_mask = np.zeros_like(m)
    cv2.fillPoly(smooth_mask, [smooth_pts], 255)

    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smooth_mask = cv2.dilate(smooth_mask, kernel_dil, iterations=inflate_iters)

    blur = cv2.GaussianBlur(smooth_mask, (7, 7), 0)
    _, smooth_mask2 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    return smooth_mask2


def smooth_mask_yolo(mask, blur_ksize=7, morph_iters=2):
    m = mask.copy().astype(np.uint8)
    if m.max() == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if morph_iters > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=morph_iters)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m_blur = cv2.GaussianBlur(m, (blur_ksize, blur_ksize), 0)
    _, m_bin = cv2.threshold(m_blur, 127, 255, cv2.THRESH_BINARY)
    return m_bin


def filter_overlaps(boxes, confs, iou_thr=0.9):
    boxes = boxes.astype(float)
    confs = confs.astype(float)
    keep = []
    idxs = np.argsort(-confs)
    taken = np.zeros(len(boxes), dtype=bool)
    for i in idxs:
        if taken[i]:
            continue
        keep.append(i)
        x1, y1, x2, y2 = boxes[i]
        area_i = (x2 - x1 + 1) * (y2 - y1 + 1)
        for j in idxs:
            if taken[j] or j == i:
                continue
            xx1 = max(x1, boxes[j][0])
            yy1 = max(y1, boxes[j][1])
            xx2 = min(x2, boxes[j][2])
            yy2 = min(y2, boxes[j][3])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            inter = w * h
            area_j = (boxes[j][2] - boxes[j][0] + 1) * (boxes[j][3] - boxes[j][1] + 1)
            iou = inter / float(area_i + area_j - inter + 1e-6)
            if iou >= iou_thr:
                taken[j] = True
    return np.array(keep, dtype=int)


# ==========================
# 5. МОДЕЛИ YOLO / КЛАССЫ
# ==========================
# Путь к YOLO весам — как в ноутбуке
BEST_WEIGHTS = "/home/kuril/runs/segment/runs_yolo11_seg/egg_tomato_yolo11s_seg2/weights/best.pt"
assert Path(BEST_WEIGHTS).exists(), f"YOLO weights not found: {BEST_WEIGHTS}"

yolo_model = YOLO(BEST_WEIGHTS)
CLASS_NAMES_YOLO = {0: "egg", 1: "tomato"}

BLUE  = (255, 0, 0)   # яйца (BGR)
GREEN = (0, 255, 0)   # красные помидоры (BGR)
PINK  = (255, 0, 255) # жёлтые помидоры (BGR)


# ==========================
# 6. ПАЙПЛАЙН process_image_bgr (как в ноутбуке)
# ==========================
def process_image_bgr(img_bgr):
    # 1) исходник в RGB (для красивой визуализации и цветных карт)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    # 2) копия для рисования контуров (в BGR)
    vis_bgr = img_bgr.copy()

    global_mask      = np.zeros((H, W), dtype=np.uint8)
    mask_eggs        = np.zeros((H, W), dtype=np.uint8)
    mask_red_toms    = np.zeros((H, W), dtype=np.uint8)
    mask_yellow_toms = np.zeros((H, W), dtype=np.uint8)

    egg_count = 0
    red_count = 0
    yellow_count = 0

    # YOLO по BGR (как в ноутбуке)
    res = yolo_model(img_bgr, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]

    boxes  = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    classes= res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), int)
    confs  = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
    masks  = res.masks.data.cpu().numpy() if res.masks is not None else []

    if len(boxes) > 0:
        keep = filter_overlaps(boxes, confs, iou_thr=0.9)
        boxes   = boxes[keep]
        classes = classes[keep]
        confs   = confs[keep]

    for det_id, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = img_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        # логрег по BGR
        feats = compute_features_rich_bgr(crop_bgr)
        label_lr = classify_logreg_3class(feats)

        # сегментация
        if cls_id == 1:  # tomato
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            best_mask, _, _ = best_tomato_mask(crop_rgb)
            if best_mask is None or best_mask.max() == 0:
                continue

            best_mask_filled = fill_all_inner_holes(best_mask)
            smooth_mask_t = smooth_contour_round(
                best_mask_filled, smooth_factor=0.4, interp_points=700, inflate_iters=3
            )
            smooth_mask_bin = (smooth_mask_t > 0).astype(np.uint8) * 255

            mask_full = np.zeros((H, W), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = smooth_mask_bin
        else:  # egg по YOLO-маске
            if len(masks) == 0 or det_id >= len(masks):
                continue
            mask_small = masks[det_id]
            mask_resized = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
            mask_full = smooth_mask_yolo(mask_bin, blur_ksize=7, morph_iters=2)

        global_mask = cv2.bitwise_or(global_mask, mask_full)
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if label_lr == "egg":
            egg_count += 1
            mask_eggs = cv2.bitwise_or(mask_eggs, mask_full)
            cv2.drawContours(vis_bgr, contours, -1, BLUE, 2)
        elif label_lr == "red_tomato":
            red_count += 1
            mask_red_toms = cv2.bitwise_or(mask_red_toms, mask_full)
            cv2.drawContours(vis_bgr, contours, -1, GREEN, 2)
        else:
            yellow_count += 1
            mask_yellow_toms = cv2.bitwise_or(mask_yellow_toms, mask_full)
            cv2.drawContours(vis_bgr, contours, -1, PINK, 2)

    # 3) всё, что уходит на фронт, в RGB
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    global_mask_final = cv2.bitwise_or(
        mask_eggs,
        cv2.bitwise_or(mask_red_toms, mask_yellow_toms)
    )

    bg_color = (100, 160, 100)
    bg_eggs   = np.full_like(img_rgb, bg_color, dtype=np.uint8)
    bg_red    = np.full_like(img_rgb, bg_color, dtype=np.uint8)
    bg_yellow = np.full_like(img_rgb, bg_color, dtype=np.uint8)

    colored_all = cv2.bitwise_and(img_rgb, img_rgb, mask=global_mask_final)

    colored_eggs = bg_eggs.copy()
    colored_eggs[mask_eggs > 0] = img_rgb[mask_eggs > 0]

    colored_red_toms = bg_red.copy()
    colored_red_toms[mask_red_toms > 0] = img_rgb[mask_red_toms > 0]

    colored_yellow_toms = bg_yellow.copy()
    colored_yellow_toms[mask_yellow_toms > 0] = img_rgb[mask_yellow_toms > 0]

    return {
        "counts": {
            "eggs": int(egg_count),
            "red": int(red_count),
            "yellow": int(yellow_count),
            "total": int(egg_count + red_count + yellow_count),
        },
        "vis": vis_rgb,               # RGB
        "all": colored_all,           # RGB
        "bin": global_mask_final,     # 1 канал
        "eggs": colored_eggs,         # RGB
        "red": colored_red_toms,      # RGB
        "yellow": colored_yellow_toms # RGB
    }



# ==========================
# 7. БУКВА (build_letter_mask_from_points + Tesseract)
# ==========================
def build_letter_mask_from_points(binary_mask,
                                  k_neighbors=3,
                                  max_dist=170,
                                  alpha=0.6,
                                  max_turn_deg=140,
                                  line_thickness=15,
                                  morph_kernel_size=7,
                                  morph_iters=1):
    """
    binary_mask: 2D np.uint8, белые точки (объекты) на чёрном фоне (0/255).
    Возвращает:
        combined   – итоговая маска буквы,
        mask_lines – только нарисованные линии,
        strokes    – список штрихов (список списков индексов точек),
        points     – массив центров объектов (N, 2) в float32.
    """
    _, bin_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    h, w = bin_mask.shape[:2]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)

    points = []
    for i in range(1, num_labels):  # 0 — фон
        cx, cy = centroids[i]
        points.append((float(cx), float(cy)))
    points = np.array(points, dtype=np.float32)
    n = len(points)

    if n < 2:
        raise RuntimeError("Мало точек для обхода")

    def build_strokes(points, k_neighbors, max_dist, alpha, max_turn_deg):
        unused = set(range(len(points)))
        strokes = []

        while unused:
            start = min(unused, key=lambda i: points[i][0] + points[i][1])
            stroke = [start]
            unused.remove(start)
            last_dir = None

            while True:
                i = stroke[-1]
                dists = np.linalg.norm(points - points[i], axis=1)
                dists[list(set(range(len(points))) - unused - {i})] = np.inf

                order = np.argsort(dists)
                candidates = [j for j in order[:k_neighbors] if j in unused]

                best_j = None
                best_cost = np.inf

                for j in candidates:
                    d = dists[j]
                    if d == np.inf or d > max_dist:
                        continue

                    dir_vec = points[j] - points[i]
                    dir_norm = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)

                    if last_dir is None:
                        angle_penalty = 0.0
                    else:
                        cos_ang = np.clip(np.dot(last_dir, dir_norm), -1.0, 1.0)
                        ang = np.degrees(np.arccos(cos_ang))
                        if ang > max_turn_deg:
                            continue
                        angle_penalty = ang / max_turn_deg

                    cost = alpha * d + (1 - alpha) * (angle_penalty * max_dist)

                    if cost < best_cost:
                        best_cost = cost
                        best_j = j

                if best_j is None:
                    break

                stroke.append(best_j)
                unused.remove(best_j)
                last_dir = (points[best_j] - points[i])
                last_dir /= (np.linalg.norm(last_dir) + 1e-6)

            strokes.append(stroke)

        return strokes

    strokes = build_strokes(points, k_neighbors, max_dist, alpha, max_turn_deg)

    mask_lines = np.zeros_like(bin_mask, dtype=np.uint8)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pts = points[stroke].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask_lines, [pts], False, 255, thickness=line_thickness)

    combined = cv2.bitwise_or(bin_mask, mask_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                kernel, iterations=morph_iters)

    return combined, mask_lines, strokes, points

def predict_letter_tesseract(combined_mask,
                             pad_ratio=0.3,
                             out_size=80,
                             psm=10):
    ys, xs = np.where(combined_mask > 0)
    if len(ys) == 0:
        return ""
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    letter_crop = combined_mask[y1:y2+1, x1:x2+1]

    h, w = letter_crop.shape
    size = max(h, w)
    pad = int(size * pad_ratio)

    square = np.zeros((size + 2*pad, size + 2*pad), dtype=np.uint8)
    y_off = pad + (size - h) // 2
    x_off = pad + (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = letter_crop

    letter_ocr = cv2.resize(square, (out_size, out_size),
                            interpolation=cv2.INTER_NEAREST)

    letter_ocr = cv2.bitwise_not(letter_ocr)

    config = rf"-l eng --oem 3 --psm {psm}"
    text = pytesseract.image_to_string(letter_ocr, config=config)

    return text.strip()


# ==========================
# 8. УТИЛИТЫ: base64 PNG
# ==========================
def img_to_b64_png(img):
    """
    Принимаем то, что есть (BGR или RGB), но перед кодированием гарантированно
    переводим в RGB-порядок каналов, как ожидает браузер.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        # OpenCV обычно хранит в BGR, но если уже RGB — лишний swap не сломает картинку,
        # только поменяет местами каналы, поэтому ДЕЛАЕМ ЯВНО BGR->RGB, а выше не трогаем.
        img_to_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Одноканальный (маски etc.) сохраняем как есть
        img_to_save = img

    ok, buf = cv2.imencode(".png", img_to_save)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")




# ==========================
# 9. FLASK + ПУТИ
# ==========================
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train_images"
SESSION_DIR = BASE_DIR / "session_images"
SESSION_DIR.mkdir(exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    if TRAIN_DIR.exists():
        files = [f.name for f in TRAIN_DIR.iterdir()
                 if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        def extract_num(name):
            digits = "".join(ch for ch in name if ch.isdigit())
            return int(digits) if digits else 0

        files = sorted(files, key=extract_num)
    else:
        files = []

    error_message = request.args.get("error")  # если когда-нибудь захочешь через query

    return render_template("index.html", train_files=files, error_message=error_message)


@app.route("/process", methods=["POST"])
def process():
    action = request.form.get("action")  # 'segment', 'letter', 'interactive'

    img_bgr = None
    source_name = None

    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        source_name = f.filename
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        selected = request.form.get("train_file")
        if selected:
            img_path = TRAIN_DIR / selected
            if img_path.exists():
                img_bgr = cv2.imread(str(img_path))
                source_name = selected

    if img_bgr is None:
        if TRAIN_DIR.exists():
            files = [f.name for f in TRAIN_DIR.iterdir()
                     if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

            def extract_num(name):
                digits = "".join(ch for ch in name if ch.isdigit())
                return int(digits) if digits else 0

            files = sorted(files, key=extract_num)
        else:
            files = []


        return render_template(
            "index.html",
            train_files=files,
            error_message="Не выбрано изображение и не загружен файл"
        )

    if action == "interactive":
        ts = int(time.time() * 1000)
        session_filename = f"img_{ts}.jpg"
        cv2.imwrite(str(SESSION_DIR / session_filename), img_bgr)

        return render_template(
            "interactive.html",
            session_filename=session_filename,
            source_name=source_name,
            original_b64=img_to_b64_png(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        )

    res = process_image_bgr(img_bgr)

    images_b64 = {
        "vis": img_to_b64_png(res["vis"]),     # RGB
        "all": img_to_b64_png(res["all"]),     # RGB
        "bin": img_to_b64_png(res["bin"]),     # 1 канал
        "eggs": img_to_b64_png(res["eggs"]),   # RGB
        "red": img_to_b64_png(res["red"]),     # RGB
        "yellow": img_to_b64_png(res["yellow"])# RGB
    }


    letter_points_b64 = None
    letter_text = ""
    letter_mask_b64 = None

    if action == "letter":
        global_mask = res["bin"]
        try:
            combined, _, _, points = build_letter_mask_from_points(global_mask)
            letter_text = predict_letter_tesseract(combined)
            letter_mask_b64 = img_to_b64_png(combined)

            # рисуем исходные точки на оригинальном изображении
            # берём RGB-картинку "vis" (или img_rgb, если хочешь без контуров)
            vis_for_points = res["vis"].copy()
            for (cx, cy) in points:
                cv2.circle(vis_for_points, (int(cx), int(cy)), 6, (255, 0, 0), -1)  # синие точки

            letter_points_b64 = img_to_b64_png(vis_for_points)
        except Exception:
            letter_text = ""
            letter_mask_b64 = None
            letter_points_b64 = None


    return render_template(
        "result.html",
        action=action,
        source_name=source_name,
        counts=res["counts"],
        images=images_b64,
        letter_text=letter_text,
        letter_mask_b64=letter_mask_b64,
        letter_points_b64=letter_points_b64,
    )



@app.route("/api/interactive", methods=["POST"])
def api_interactive():
    data = request.json
    filename = data.get("filename")

    blur_val = int(data.get("blur", 1))
    bright_val = int(data.get("brightness", 0))
    contrast_val = float(data.get("contrast", 1.0))
    smooth_val = int(data.get("smooth", 0))
    negative = data.get("negative", False)

    img_path = SESSION_DIR / filename
    if not img_path.exists():
        return jsonify({"error": "File not found"}), 404

    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Bad image"}), 400

    if bright_val != 0 or contrast_val != 1.0:
        img = cv2.convertScaleAbs(img, alpha=contrast_val, beta=bright_val)

    if negative:
        img = cv2.bitwise_not(img)

    if blur_val % 2 == 0:
        blur_val += 1
    if blur_val > 1:
        img = cv2.GaussianBlur(img, (blur_val, blur_val), 0)

    if smooth_val > 0:
        img = cv2.bilateralFilter(img, 9, smooth_val, smooth_val)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return jsonify({
        "result_b64": img_to_b64_png(img_rgb)
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
