from argparse import ArgumentParser
from asyncio import staggered
import json
from pathlib import Path
from re import I

import cv2
import numpy as np
import torch
from torch import Tensor

from utils.constants import colors
from utils.misc import minmax_normalize


def visualize(image, mask=None, bboxes=[], name=None, labels=None, get_other=None, mkup_dir=None, W=1440, H=810):
    X, Y, Z, dx, dy, dz = 0, 0, 0, 0, 0, 0
    aw, ah, ax, ay = 0, 0, 0, 0
    cw, ch, cx, cy = 0, 0, 0, 0
    sw, sh, sx, sy = 0, 0, 0, 0
    segmentation = False
    focus = "none"
    cross = "cross"
    corner = None
    bbox_id = None
    cross_opts = ["none", "cross", "box", "point"]
    box_size = 10
    boxes_saved = -1

    mkup_bboxes = []

    def reset():
        nonlocal boxes_saved
        nonlocal mkup_bboxes
        nonlocal X, Y, Z, dx, dy, dz
        nonlocal aw, ah, ax, ay
        nonlocal cw, ch, cx, cy
        nonlocal sw, sh, sx, sy
        nonlocal segmentation
        nonlocal image
        nonlocal mask

        boxes_saved = -1
        mkup_bboxes = []

        if mask is not None:
            if isinstance(mask, Tensor):
                mask = mask.detach().cpu().numpy()
                
            mask = mask.astype(int)
            mask = colors[mask] / 255

        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()

        image = minmax_normalize(image, inplace=False)

        X = image.shape[0] // 2
        Y = image.shape[1] // 2
        Z = image.shape[2] // 2

        w = W // 3
        
        dx, dy, dz = image.shape
        aw, ah = min(w, dx * H // dy), min(dy * w // dx, H)
        ax, ay = 0, (H - ah) // 2

        cw, ch = min(w, dx * H // dz), min(dz * w // dx, H)
        cx, cy = w, (H - ch) // 2

        sw, sh = min(w, dy * H // dz), min(dz * w // dy, H)
        sx, sy = w * 2, (H - sh) // 2

        segmentation = mask is not None

    def redraw_canvas():
        canvas = np.zeros((H, W, 3))
        w, h, d = image.shape

        axial = image[:, :, d-1-Z][:, :, None].repeat(3, -1)
        coronal = image[:, h-1-Y, :][:, :, None].repeat(3, -1)
        sagittal = image[X, :, :][:, :, None].repeat(3, -1)

        if bboxes or mkup_bboxes:
            for bbox in (bboxes + mkup_bboxes):
                x1, y1, z1, x2, y2, z2 = map(int, bbox)
                
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                if z1 > z2: z1, z2 = z2, z1

                if z1 <= d - 1 - Z <= z2:
                    cv2.rectangle(axial, (y1, x1), (y2, x2), (0, 0, 1), 2)

                if y1 <= h - 1 - Y <= y2:
                    cv2.rectangle(coronal, (z1, x1), (z2, x2), (0, 0, 1), 2)

                if x1 <= X <= x2:
                    cv2.rectangle(sagittal, (z1, y1), (z2, y2), (0, 0, 1), 2)

        canvas[ay:ay+ah, ax:ax+aw] = cv2.resize(axial[:, ::-1].swapaxes(0, 1), (aw, ah))
        canvas[cy:cy+ch, cx:cx+cw] = cv2.resize(coronal[:, ::-1].swapaxes(0, 1), (cw, ch))
        canvas[sy:sy+sh, sx:sx+sw] = cv2.resize(sagittal[::-1, ::-1].swapaxes(0, 1), (sw, sh))

        if segmentation and mask is not None:
            canvas = canvas * 0.7
            canvas[ay:ay+ah, ax:ax+aw] += cv2.resize(mask[:, ::-1, d-1-Z].swapaxes(0, 1), (aw, ah)) * 0.3
            canvas[cy:cy+ch, cx:cx+cw] += cv2.resize(mask[:, h-1-Y, ::-1].swapaxes(0, 1), (cw, ch)) * 0.3
            canvas[sy:sy+sh, sx:sx+sw] += cv2.resize(mask[X, ::-1, ::-1].swapaxes(0, 1), (sw, sh)) * 0.3

        x = int(ax + (X + 0.5) / dx * aw)
        y = int(ay + (Y + 0.5) / dy * ah)

        if cross == "cross":
            cv2.line(canvas, (x, ay), (x, ay+ah), (1, 0, 0))
            cv2.line(canvas, (ax, y), (ax+aw, y), (0, 1, 0))
        elif cross == "point":
            cv2.drawMarker(canvas, (x, y), color=(0, 0, 1), markerSize=1)
        elif cross == "box":
            x1, y1 = x - box_size // 2, y - box_size // 2
            x2, y2 = x + box_size // 2, y + box_size // 2
            
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 0, 1), thickness=1)

        x = int(cx + (X + 0.5) / dx * cw)
        y = int(cy + (Z + 0.5) / dz * ch)

        if cross == "cross":
            cv2.line(canvas, (x, cy), (x, cy+ch), (1, 0, 0))
            cv2.line(canvas, (cx, y), (cx+cw, y), (0, 0, 1))
        elif cross == "point":
            cv2.drawMarker(canvas, (x, y), color=(0, 1, 0), markerSize=1)
        elif cross == "box":
            x1, y1 = x - box_size // 2, y - box_size // 2
            x2, y2 = x + box_size // 2, y + box_size // 2
            
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 1, 0), thickness=1)

        x = int(sx + (Y + 0.5) / dy * sw)
        y = int(sy + (Z + 0.5) / dz * sh)

        if cross == "cross":
            cv2.line(canvas, (x, sy), (x, sy+sh), (0, 1, 0))
            cv2.line(canvas, (sx, y), (sx+sw, y), (0, 0, 1))
        elif cross == "point":
            cv2.drawMarker(canvas, (x, y), color=(1, 0, 0), markerSize=1)
        elif cross == "box":
            x1, y1 = x - box_size // 2, y - box_size // 2
            x2, y2 = x + box_size // 2, y + box_size // 2
            
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(1, 0, 0), thickness=1)

        if name is not None:
            cv2.putText(canvas, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1))
        if labels is not None:
            cv2.putText(canvas, str(labels), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1))
        if boxes_saved != -1:
            cv2.putText(canvas, f"saved {boxes_saved} boxes", (W - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1))

        return canvas

    def mouse_event(event, x, y, flags, param):
        nonlocal focus
        nonlocal corner, bbox_id
        nonlocal Z, Y, X

        w, h, d = image.shape

        if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDOWN]:
            if 0 < x < W // 3:
                focus = "axial"
            elif W//3 < x < W // 3 * 2:
                focus = "coronal"
            elif W//3 * 2 < x < W:
                focus = "sagittal"

        if event in [cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            if focus == "axial" and ax < x < ax + aw and ay < y < ay + ah:
                X = int((x - ax) / aw * dx)
                Y = int((y - ay) / ah * dy)
            elif focus == "coronal" and cx < x < cx + cw and cy < y < cy + ch:
                X = int((x - cx) / cw * dx)
                Z = int((y - cy) / ch * dz)
            elif focus == "sagittal" and sx < x < sx + sw and sy < y < sy + sh:
                Y = int((x - sx) / sw * dy)
                Z = int((y - sy) / sh * dz)

            if bbox_id is not None:
                bbox = mkup_bboxes[bbox_id]
                bbox[corner] = [X, h - 1 - Y, d - 1 - Z]

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(mkup_bboxes):
                min_dist = 10
                for a in [0, 3]:
                    for b in [1, 4]:
                        for c in [2, 5]:
                            dist = np.linalg.norm(box[[a, b, c]] - [X, h - 1 - Y, d - 1 - Z])
                            if dist < min_dist:
                                min_dist = dist
                                bbox_id = i
                                corner = [a, b, c]

            if bbox_id is None:
                bbox_id = len(mkup_bboxes)
                mkup_bboxes.append(np.array([X, h - 1 - Y, d - 1 - Z, X, h - 1 - Y, d - 1 - Z]))
                corner = [0, 1, 2]

        if event in [cv2.EVENT_RBUTTONUP, cv2.EVENT_LBUTTONUP]:
            focus = "none"
            bbox_id = None

    cv2.namedWindow("visualization")
    cv2.resizeWindow("visualization", W, H)
    cv2.setMouseCallback("visualization", mouse_event)

    reset()

    while True:
        canvas = redraw_canvas()
        cv2.imshow("visualization", canvas)

        key = cv2.waitKey(1)
        if key == 27: # esc
            break

        if key == ord(" ") and mask is not None:
            segmentation = not segmentation

        if key == ord("x"):
            cross = cross_opts[(cross_opts.index(cross) + 1) % len(cross_opts)]

        if key == ord(".") and get_other is not None:
            image, mask, name, labels = get_other(+1)
            reset()

        if key == ord(",")  and get_other is not None:
            image, mask, name, labels = get_other(-1)
            reset()

        if key == 13 and mkup_dir is not None: # enter
            for bbox in mkup_bboxes:
                ul = bbox[:3]
                br = bbox[3:]
                ul_ = np.minimum(ul, br)
                br_ = np.maximum(ul, br)

                bbox[:3] = ul_
                bbox[3:] = br_

            (mkup_dir / f"{name}.json").write_text(json.dumps(
                {name: [bbox.tolist() for bbox in mkup_bboxes]}
            ))

            boxes_saved = len(mkup_bboxes)

            print(f"Saved {len(mkup_bboxes)} bboxes in {name} image")

        if key == ord("z"):
            mkup_bboxes.pop()

        if key == 8: # backspace
            mkup_bboxes = []

    cv2.destroyAllWindows()
