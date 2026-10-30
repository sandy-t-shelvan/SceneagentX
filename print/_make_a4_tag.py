"""One-off: A4 vector page with a 50 mm tag36h11 and centimetre marks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
MM = 72.0 / 25.4
PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
TAG_MM = 50.0
QUIET_MM = 18.0
RULER_MM = 28.0  # dimension line sits this far from the black square

HELV = {
    " ": 278, ".": 278, "%": 889, "0": 556, "1": 556, "2": 556, "3": 556,
    "4": 556, "5": 556, "6": 556, "7": 556, "8": 556, "9": 556,
    "A": 667, "B": 667, "D": 722, "E": 611, "F": 611, "I": 278, "M": 833,
    "N": 722, "O": 778, "P": 611, "R": 667, "S": 556, "T": 611, "a": 556,
    "b": 556, "c": 444, "d": 556, "e": 556, "f": 278, "g": 556, "h": 556,
    "i": 278, "l": 278, "m": 833, "n": 556, "o": 556, "p": 556, "q": 556,
    "r": 333, "s": 500, "t": 278, "u": 556, "x": 500, "z": 500,
}


def text_width(text: str, size: float) -> float:
    return sum(HELV.get(ch, 560) for ch in text) / 1000.0 * size


def tag_grid() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    modules = int(dictionary.markerSize) + 2
    cell = 20
    img = cv2.aruco.generateImageMarker(dictionary, 0, modules * cell, borderBits=1)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grid = img.reshape(modules, cell, modules, cell)[:, cell // 2, :, cell // 2] < 128
    return grid


def build_pdf(grid: np.ndarray) -> bytes:
    modules = grid.shape[0]
    page_w = PAGE_W_MM * MM
    page_h = PAGE_H_MM * MM
    tag = TAG_MM * MM
    module = tag / modules
    tag_x = (PAGE_W_MM - TAG_MM) / 2.0 * MM
    tag_y = 128.0 * MM
    cmds: list[str] = [
        "1 g",
        f"0 0 {page_w:.4f} {page_h:.4f} re f",
        "0 g",
    ]

    eps = 0.04
    for r in range(modules):
        for c in range(modules):
            if not bool(grid[r, c]):
                continue
            x0 = tag_x + c * module - (eps if c else 0.0)
            x1 = tag_x + (c + 1) * module + (eps if c < modules - 1 else 0.0)
            y1 = tag_y + (modules - r) * module + (eps if r else 0.0)
            y0 = tag_y + (modules - r - 1) * module - (eps if r < modules - 1 else 0.0)
            cmds.append(f"{x0:.4f} {y0:.4f} {x1 - x0:.4f} {y1 - y0:.4f} re f")

    def line(x0: float, y0: float, x1: float, y1: float) -> None:
        cmds.append(f"{x0:.4f} {y0:.4f} m {x1:.4f} {y1:.4f} l S")

    cmds.append("1.15 w 0 J 0 j")
    h_y = tag_y - RULER_MM * MM
    v_x = tag_x - RULER_MM * MM
    ext0 = QUIET_MM * MM
    tick = 3.2 * MM
    end_tick = 5.0 * MM

    # Extension lines stay outside the white quiet zone.
    line(tag_x, tag_y - ext0, tag_x, h_y - 2.2 * MM)
    line(tag_x + tag, tag_y - ext0, tag_x + tag, h_y - 2.2 * MM)
    line(tag_x - ext0, tag_y, v_x - 2.2 * MM, tag_y)
    line(tag_x - ext0, tag_y + tag, v_x - 2.2 * MM, tag_y + tag)

    line(tag_x, h_y, tag_x + tag, h_y)
    line(v_x, tag_y, v_x, tag_y + tag)
    for i in range(6):
        height = end_tick if i in (0, 5) else tick
        x = tag_x + i * 10.0 * MM
        y = tag_y + i * 10.0 * MM
        line(x, h_y, x, h_y + height)
        line(v_x, y, v_x + height, y)

    def label(text: str, x: float, y: float, size: float) -> None:
        cmds.append(f"BT /F1 {size:.2f} Tf {x:.3f} {y:.3f} Td ({text}) Tj ET")

    def centered(text: str, cx: float, baseline: float, size: float) -> None:
        label(text, cx - text_width(text, size) / 2.0, baseline, size)

    for i in range(6):
        name = str(i)
        centered(name, tag_x + i * 10.0 * MM, h_y - 14, 9)
        label(name, v_x - 7 - text_width(name, 9), tag_y + i * 10.0 * MM - 3.2, 9)
    centered("5 cm", tag_x + tag / 2, h_y - 32, 16)
    side = "5 cm"
    label(side, v_x - 22 - text_width(side, 16), tag_y + tag / 2 - 6, 16)

    centered("tag36h11    id 0", page_w / 2, 268 * MM, 18)
    centered("Black square = 5.0 cm x 5.0 cm", page_w / 2, 256 * MM, 14)
    centered("Print at 100% actual size. Do not fit to page.", page_w / 2, 246 * MM, 11)
    centered("After printing, the black square must measure 50 mm on each side.", page_w / 2, 16 * MM, 10)

    stream = ("\n".join(cmds) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
            f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ).encode("ascii"),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_at = sum(len(part) for part in chunks)
    xref = [f"xref\n0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return b"".join(chunks + xref + [trailer])


def build_preview(grid: np.ndarray, dpi: float = 130.0) -> Image.Image:
    def px(mm: float) -> int:
        return int(round(mm / 25.4 * dpi))

    image = Image.new("L", (px(PAGE_W_MM), px(PAGE_H_MM)), 255)
    draw = ImageDraw.Draw(image)
    modules = grid.shape[0]
    cell = TAG_MM / modules
    tag_x = (PAGE_W_MM - TAG_MM) / 2.0
    tag_y_top = PAGE_H_MM - (118.0 + TAG_MM)
    for r in range(modules):
        for c in range(modules):
            if not bool(grid[r, c]):
                continue
            x0 = px(tag_x + c * cell)
            y0 = px(tag_y_top + r * cell)
            x1 = px(tag_x + (c + 1) * cell) - 1
            y1 = px(tag_y_top + (r + 1) * cell) - 1
            draw.rectangle((x0, y0, x1, y1), fill=0)

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    def font(pt: float) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(font_path), max(int(round(pt / 72 * dpi)), 8))

    tag_bottom = 128.0
    tag_left = tag_x
    h_y = tag_bottom - RULER_MM
    v_x = tag_left - RULER_MM

    def vline(x_mm: float, y0_mm: float, y1_mm: float, width: int = 2) -> None:
        draw.line((px(x_mm), px(PAGE_H_MM - y0_mm), px(x_mm), px(PAGE_H_MM - y1_mm)), fill=0, width=width)

    def hline(y_mm: float, x0_mm: float, x1_mm: float, width: int = 2) -> None:
        draw.line((px(x0_mm), px(PAGE_H_MM - y_mm), px(x1_mm), px(PAGE_H_MM - y_mm)), fill=0, width=width)

    vline(tag_left, tag_bottom - QUIET_MM, h_y - 2.2)
    vline(tag_left + TAG_MM, tag_bottom - QUIET_MM, h_y - 2.2)
    hline(tag_bottom, tag_left - QUIET_MM, v_x - 2.2)
    hline(tag_bottom + TAG_MM, tag_left - QUIET_MM, v_x - 2.2)
    hline(h_y, tag_left, tag_left + TAG_MM)
    vline(v_x, tag_bottom, tag_bottom + TAG_MM)
    for i in range(6):
        height = 5.0 if i in (0, 5) else 3.2
        hline_x = tag_left + i * 10.0
        vline(hline_x, h_y, h_y + height, 2)
        hline(tag_bottom + i * 10.0, v_x, v_x + height, 2)
        name = str(i)
        fnt = font(9)
        box = draw.textbbox((0, 0), name, font=fnt)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((px(hline_x) - tw / 2, px(PAGE_H_MM - (h_y - 3.5))), name, fill=0, font=fnt)
        draw.text((px(v_x - 2.4) - tw, px(PAGE_H_MM - (tag_bottom + i * 10.0)) - th / 2), name, fill=0, font=fnt)
    big = font(16)
    caption = "5 cm"
    box = draw.textbbox((0, 0), caption, font=big)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text((px(tag_left + TAG_MM / 2) - tw / 2, px(PAGE_H_MM - (h_y - 9))), caption, fill=0, font=big)
    draw.text((px(v_x - 16) - tw, px(PAGE_H_MM - (tag_bottom + TAG_MM / 2)) - th / 2), caption, fill=0, font=big)

    def center(text: str, y_mm: float, pt: float) -> None:
        fnt = font(pt)
        box = draw.textbbox((0, 0), text, font=fnt)
        tw = box[2] - box[0]
        draw.text(((image.width - tw) / 2, px(PAGE_H_MM - y_mm) - box[3]), text, fill=0, font=fnt)

    center("tag36h11    id 0", 268, 18)
    center("Black square = 5.0 cm x 5.0 cm", 256, 14)
    center("Print at 100% actual size. Do not fit to page.", 246, 11)
    center("After printing, the black square must measure 50 mm on each side.", 16, 10)
    return image


def main() -> None:
    grid = tag_grid()
    pdf_path = OUT / "tag36h11_id0_A4.pdf"
    pdf_path.write_bytes(build_pdf(grid))
    preview = build_preview(grid)
    preview_path = OUT / "_a4_preview.png"
    preview.save(preview_path)
    gray = np.array(preview)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _corners, ids, _rejected = detector.detectMarkers(gray)
    found = None if ids is None else int(np.asarray(ids).flatten()[0])
    if found != 0 or (ids is not None and len(np.asarray(ids).flatten()) != 1):
        raise SystemExit(f"preview detection failed: {ids}")
    print(f"pdf {pdf_path} bytes {pdf_path.stat().st_size}")
    print(f"preview {preview_path} detected id {found}")


if __name__ == "__main__":
    main()
