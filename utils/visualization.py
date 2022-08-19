from asyncio import staggered
from re import I
import torch
import cv2
from pathlib import Path
import numpy as np

from argparse import ArgumentParser

from utils.constants import colors


def visualize(image, mask=None, name=None, get_other=None, W=1440, H=810):
    X, Y, Z, dx, dy, dz = 0, 0, 0, 0, 0, 0
    aw, ah, ax, ay = 0, 0, 0, 0
    cw, ch, cx, cy = 0, 0, 0, 0
    sw, sh, sx, sy = 0, 0, 0, 0
    segmentation = False
    focus = "none"

    def reset():
        nonlocal X, Y, Z, dx, dy, dz
        nonlocal aw, ah, ax, ay
        nonlocal cw, ch, cx, cy
        nonlocal sw, sh, sx, sy
        nonlocal segmentation
        nonlocal mask

        if mask is not None:
            mask = colors[mask] / 255

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

        canvas[ay:ay+ah, ax:ax+aw] = cv2.resize(image[:, ::-1, -Z].T, (aw, ah))[:, :, None]
        canvas[cy:cy+ch, cx:cx+cw] = cv2.resize(image[:, -Y, ::-1].T, (cw, ch))[:, :, None]
        canvas[sy:sy+sh, sx:sx+sw] = cv2.resize(image[X, ::-1, ::-1].T, (sw, sh))[:, :, None]

        if segmentation and mask is not None:
            canvas = canvas * 0.7
            canvas[ay:ay+ah, ax:ax+aw] += cv2.resize(mask[:, ::-1, -Z].swapaxes(0, 1), (aw, ah)) * 0.3
            canvas[cy:cy+ch, cx:cx+cw] += cv2.resize(mask[:, -Y, ::-1].swapaxes(0, 1), (cw, ch)) * 0.3
            canvas[sy:sy+sh, sx:sx+sw] += cv2.resize(mask[X, ::-1, ::-1].swapaxes(0, 1), (sw, sh)) * 0.3

        x = int(ax + X / dx * aw)
        cv2.line(canvas, (x, ay), (x, ay+ah), (1, 0, 0))
        y = int(ay + Y / dy * ah)
        cv2.line(canvas, (ax, y), (ax+aw, y), (0, 1, 0))

        x = int(cx + X / dx * cw)
        cv2.line(canvas, (x, cy), (x, cy+ch), (1, 0, 0))
        y = int(cy + Z / dz * ch)
        cv2.line(canvas, (cx, y), (cx+cw, y), (0, 0, 1))

        x = int(sx + Y / dy * sw)
        cv2.line(canvas, (x, sy), (x, sy+sh), (0, 1, 0))
        y = int(sy + Z / dz * sh)
        cv2.line(canvas, (sx, y), (sx+sw, y), (0, 0, 1))

        if name is not None:
            cv2.putText(canvas, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1))

        return canvas

    def mouse_event(event, x, y, flags, param):
        nonlocal focus
        nonlocal Z, Y, X

        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 < x < W // 3:
                focus = "axial"
            elif W//3 < x < W // 3 * 2:
                focus = "coronal"
            elif W//3 * 2 < x < W:
                focus = "sagittal"

        elif event == cv2.EVENT_LBUTTONUP:
            focus = "none"

        elif event == cv2.EVENT_MOUSEMOVE:
            if focus == "axial" and ax < x < ax + aw and ay < y < ay + ah:
                X = int((x - ax) / aw * dx)
                Y = int((y - ay) / ah * dy)
            elif focus == "coronal" and cx < x < cx + cw and cy < y < cy + ch:
                X = int((x - cx) / cw * dx)
                Z = int((y - cy) / ch * dz)
            elif focus == "sagittal" and sx < x < sx + sw and sy < y < sy + sh:
                Y = int((x - sx) / sw * dy)
                Z = int((y - sy) / sh * dz)

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

        if key == ord(".") and get_other is not None:
            image, mask, name = get_other(+1)
            reset()

        if key == ord(",")  and get_other is not None:
            image, mask, name = get_other(-1)
            reset()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", nargs=1, default="./preprocessed_data")

    args = parser.parse_args()

    root = Path(args.dir[0])

    images = list(root.glob("images3d/*"))
    masks = list(root.glob("segmentations/*"))

    i = 0
    def get_other(diff):
        global i
        i = (i + diff) % len(images)

        image = torch.load(images[i]).numpy()
        mask = torch.load(masks[i]).numpy()

        return image, mask, images[i].name
    
    image, mask, name = get_other(0)

    visualize(image, mask, name, get_other)
