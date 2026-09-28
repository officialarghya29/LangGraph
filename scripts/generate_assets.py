#!/usr/bin/env python3
"""Generate the README image assets.

Every PNG under ``docs/assets/`` is produced by this script, so the artwork
lives in version control as source rather than as an opaque binary that nobody
can regenerate. Re-run it after changing the palette or layout:

    python scripts/generate_assets.py

Requires Pillow (a build-time tool only; it is not a runtime dependency of the
application).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageFont

#: Anything Pillow accepts as a fill or outline for a shape.
Fill = str | int | float | tuple[int, ...] | None

# --------------------------------------------------------------------------- #
# Paths and fonts
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"
FONT_ROOT = Path("/usr/share/fonts/truetype")

#: Supersampling factor.
#:
#: The builders below are written in *design units* — the numbers that describe
#: the layout, not pixel counts. Rendering multiplies every one of them by this
#: factor. A README image is scaled down to the width of the column it sits in
#: (roughly 830 CSS pixels on GitHub), so a 1:1 render of an 1800-pixel-wide
#: diagram shrinks a 17-pixel label to about eight CSS pixels and makes it
#: unreadable. Drawing at 2x keeps the geometry identical while leaving enough
#: pixels for text to survive that downscale sharp.
SCALE = 2

#: Resolution written into the PNG's pHYs chunk, in dots per inch. Declaring it
#: lets viewers that honour pixel density pick a sensible on-screen size rather
#: than assuming 72 dpi and rendering the asset at double size.
DPI = 144

FONT_DISPLAY = FONT_ROOT / "ubuntu" / "Ubuntu-B.ttf"
FONT_MEDIUM = FONT_ROOT / "ubuntu" / "Ubuntu-M.ttf"
FONT_REGULAR = FONT_ROOT / "ubuntu" / "Ubuntu-R.ttf"
FONT_LIGHT = FONT_ROOT / "ubuntu" / "Ubuntu-L.ttf"
FONT_MONO = FONT_ROOT / "ubuntu" / "UbuntuMono-R.ttf"
FONT_MONO_BOLD = FONT_ROOT / "ubuntu" / "UbuntuMono-B.ttf"

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

BG_TOP = (9, 12, 28)
BG_BOTTOM = (3, 5, 13)

PANEL = (19, 25, 49, 210)
PANEL_SOFT = (15, 19, 38, 170)
BORDER = (46, 58, 100)

TEXT = (233, 240, 255)
MUTED = (143, 158, 199)
DIM = (97, 111, 152)

CYAN = (34, 211, 238)
VIOLET = (139, 92, 246)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)
RED = (248, 113, 113)
PINK = (244, 114, 182)
BLUE = (96, 165, 250)


#: Every string drawn by the current asset, with its box in design units.
#: Populated by :class:`ScaledDraw` and inspected by :func:`audit_layout`, which
#: is what turns "this diagram looks crowded" into a checkable assertion.
_RECORDED_TEXT: list[tuple[tuple[float, float, float, float], str]] = []


def audit_layout(width: int, height: int) -> list[str]:
    """Return the layout problems in the most recently built asset.

    Two invariants are checked, both of which were violated somewhere in the
    first version of these diagrams:

    1. **Nothing escapes the canvas.** A label drawn past the edge is silently
       cropped, which looks like a rendering bug rather than a layout one.
    2. **No two labels overlap.** Overlapping text is always a mistake here;
       there is no case where two strings are meant to occupy the same pixels.

    Args:
        width: Canvas width in design units.
        height: Canvas height in design units.

    Returns:
        One human-readable line per problem, empty when the layout is sound.
    """
    problems: list[str] = []
    for (x0, y0, x1, y1), text in _RECORDED_TEXT:
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            box = tuple(round(value, 1) for value in (x0, y0, x1, y1))
            problems.append(f"{text!r} escapes the {width}x{height} canvas at {box}")

    for index, (first, first_text) in enumerate(_RECORDED_TEXT):
        for second, second_text in _RECORDED_TEXT[index + 1 :]:
            if (
                first[0] < second[2]
                and second[0] < first[2]
                and first[1] < second[3]
                and second[1] < first[3]
            ):
                problems.append(f"{first_text!r} overlaps {second_text!r}")
    return problems


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at the given design-unit size.

    Args:
        path: Font file to load.
        size: Size in design units; scaled by :data:`SCALE` before loading.
    """
    return ImageFont.truetype(str(path), round(size * SCALE))


class Surface(Protocol):
    """The drawing operations the layout helpers rely on.

    Declared as a protocol so a helper can accept either a real
    :class:`ImageDraw.ImageDraw` or a :class:`ScaledDraw` without either having
    to subclass the other. mypy checks calls against this shape rather than
    against a concrete class, which is what keeps the coordinate-scaling
    wrapper invisible to everything downstream of it.
    """

    def textlength(self, text: str, font: ImageFont.FreeTypeFont | None = None) -> float:
        """Return the width ``text`` would occupy."""
        ...

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        font: ImageFont.FreeTypeFont | None = None,
        fill: Fill = None,
        anchor: str | None = None,
    ) -> None:
        """Draw ``text`` anchored at ``xy``."""
        ...

    def line(self, xy: Sequence[tuple[float, float]], fill: Fill = None, width: float = 1) -> None:
        """Draw a polyline through ``xy``."""
        ...

    def ellipse(
        self, xy: Sequence[float], fill: Fill = None, outline: Fill = None, width: float = 1
    ) -> None:
        """Draw an ellipse inside the bounding box ``xy``."""
        ...

    def polygon(
        self,
        xy: Sequence[tuple[float, float]],
        fill: Fill = None,
        outline: Fill = None,
        width: float = 1,
    ) -> None:
        """Draw a polygon through the vertices ``xy``."""
        ...

    def rounded_rectangle(
        self,
        xy: Sequence[float],
        radius: float = 0,
        fill: Fill = None,
        outline: Fill = None,
        width: float = 1,
    ) -> None:
        """Draw a rounded rectangle inside the bounding box ``xy``."""
        ...


class ScaledDraw:
    """A :class:`Surface` that accepts coordinates in design units.

    Every coordinate, pen width, corner radius, and font size is multiplied by
    :data:`SCALE` on its way to the real draw object. That keeps the builders
    readable — they describe the layout, not the raster — while the render
    happens at a higher resolution. State lives only in the wrapped draw object,
    so one proxy is created per layer.

    Measurement is scaled the other way: :meth:`textlength` divides by
    :data:`SCALE` so callers keep reasoning in design units. Without that,
    every layout calculation that measures text would silently switch to pixels
    and overflow its container.
    """

    __slots__ = ("_draw",)

    def __init__(self, image: Image.Image) -> None:
        self._draw = ImageDraw.Draw(image)

    @staticmethod
    def _points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Return the points converted from design units to pixels."""
        return [(x * SCALE, y * SCALE) for x, y in points]

    @staticmethod
    def _pen(width: float) -> int:
        """Return a stroke width in pixels, never thinner than one."""
        return max(1, round(width * SCALE))

    def textlength(self, text: str, font: ImageFont.FreeTypeFont | None = None) -> float:
        """Return the rendered width of ``text`` in design units."""
        return self._draw.textlength(text, font=font) / SCALE

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        font: ImageFont.FreeTypeFont | None = None,
        fill: Fill = None,
        anchor: str | None = None,
    ) -> None:
        """Draw ``text`` at a design-unit position, recording its box for audit."""
        pixels = (xy[0] * SCALE, xy[1] * SCALE)
        self._draw.text(pixels, text, font=font, fill=fill, anchor=anchor)
        x0, y0, x1, y1 = self._draw.textbbox(pixels, text, font=font, anchor=anchor)
        _RECORDED_TEXT.append(((x0 / SCALE, y0 / SCALE, x1 / SCALE, y1 / SCALE), text))

    def line(self, xy: Sequence[tuple[float, float]], fill: Fill = None, width: float = 1) -> None:
        """Draw a polyline between design-unit points."""
        self._draw.line(self._points(xy), fill=fill, width=self._pen(width))

    def ellipse(
        self, xy: Sequence[float], fill: Fill = None, outline: Fill = None, width: float = 1
    ) -> None:
        """Draw an ellipse inside a design-unit bounding box."""
        self._draw.ellipse(
            [value * SCALE for value in xy], fill=fill, outline=outline, width=self._pen(width)
        )

    def polygon(
        self,
        xy: Sequence[tuple[float, float]],
        fill: Fill = None,
        outline: Fill = None,
        width: float = 1,
    ) -> None:
        """Draw a polygon through design-unit vertices."""
        self._draw.polygon(self._points(xy), fill=fill, outline=outline, width=self._pen(width))

    def rounded_rectangle(
        self,
        xy: Sequence[float],
        radius: float = 0,
        fill: Fill = None,
        outline: Fill = None,
        width: float = 1,
    ) -> None:
        """Draw a rounded rectangle at a design-unit box and radius."""
        self._draw.rounded_rectangle(
            [value * SCALE for value in xy],
            radius=radius * SCALE,
            fill=fill,
            outline=outline,
            width=self._pen(width),
        )


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def vertical_gradient(
    width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> Image.Image:
    """Build a vertical linear-gradient image."""
    span = max(height - 1, 1)
    column = [
        (
            round(top[0] + (bottom[0] - top[0]) * (y / span)),
            round(top[1] + (bottom[1] - top[1]) * (y / span)),
            round(top[2] + (bottom[2] - top[2]) * (y / span)),
        )
        for y in range(height)
    ]
    strip = Image.new("RGB", (1, height))
    strip.putdata(column)
    return strip.resize((width, height), Image.Resampling.BICUBIC)


def draw_on(image: Image.Image) -> ScaledDraw:
    """Return a design-unit drawing surface for ``image``."""
    return ScaledDraw(image)


def canvas(
    width: int,
    height: int,
    top: tuple[int, int, int] = BG_TOP,
    bottom: tuple[int, int, int] = BG_BOTTOM,
) -> Image.Image:
    """Create the base RGBA canvas for an asset, sized in design units."""
    return vertical_gradient(width * SCALE, height * SCALE, top, bottom).convert("RGBA")


def glow(
    base: Image.Image,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: int = 90,
    blur: int = 110,
) -> Image.Image:
    """Composite a soft radial bloom onto the canvas, positioned in design units."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center[0] * SCALE, center[1] * SCALE
    span = radius * SCALE
    draw.ellipse([cx - span, cy - span, cx + span, cy + span], fill=(*color, alpha))
    return Image.alpha_composite(base, layer.filter(ImageFilter.GaussianBlur(blur * SCALE)))


def grid(
    base: Image.Image,
    step: int = 44,
    alpha: int = 14,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Lay a faint technical grid over the canvas."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = base.size
    spacing = max(1, round(step * SCALE))
    for x in range(0, width, spacing):
        draw.line([(x, 0), (x, height)], fill=(*color, alpha), width=1)
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=(*color, alpha), width=1)
    return Image.alpha_composite(base, layer)


def text_width(draw: Surface, text: str, f: ImageFont.FreeTypeFont, spacing: float = 0.0) -> float:
    """Measure a string including letter spacing."""
    return sum(draw.textlength(c, font=f) for c in text) + spacing * max(len(text) - 1, 0)


def tracked(
    draw: Surface,
    xy: tuple[float, float],
    text: str,
    f: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    spacing: float = 0.0,
    center_x: float | None = None,
) -> float:
    """Draw text with optional letter spacing, returning the rendered width."""
    x, y = xy
    if center_x is not None:
        x = center_x - text_width(draw, text, f, spacing) / 2
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += draw.textlength(ch, font=f) + spacing
    return x - xy[0] if center_x is None else text_width(draw, text, f, spacing)


def wrap_text(
    draw: Surface,
    text: str,
    f: ImageFont.FreeTypeFont,
    max_width: float,
    max_lines: int = 2,
) -> list[str]:
    """Break ``text`` into lines that each fit within ``max_width``.

    Used instead of a hard character cut, which is what overflowed these cards
    before: a truncated string knows nothing about the font it will be drawn
    with, so a limit that looks safe in code still runs off the card. Measuring
    the actual glyphs is the only way to be sure.

    Args:
        draw: Surface used for measurement.
        text: The string to fit.
        f: Font the string will be drawn with.
        max_width: Available width, in design units.
        max_lines: Maximum lines to emit; the last is ellipsised if needed.

    Returns:
        One or more lines, each no wider than ``max_width``.
    """
    lines: list[str] = []
    current = ""
    consumed = 0
    words = text.split()
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=f) > max_width:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = candidate
        consumed += 1

    # ``current`` holds the last word accepted before the loop stopped. It is
    # dropped only when a full set of lines was already emitted and words
    # remain, which is exactly the case the ellipsis below reports.
    if len(lines) < max_lines and current:
        lines.append(current)
    elif consumed < len(words):
        lines[-1] = f"{lines[-1].rstrip()} \u2026"
    return lines


def hexagon(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    """Return the six vertices of a pointy-top hexagon."""
    return [
        (
            cx + radius * math.cos(math.radians(60 * i - 90)),
            cy + radius * math.sin(math.radians(60 * i - 90)),
        )
        for i in range(6)
    ]


def arrow(
    draw: Surface,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 3,
    head: int = 15,
) -> None:
    """Draw a straight line with an arrow head at the end."""
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for delta in (150, -150):
        a = angle + math.radians(delta)
        draw.line(
            [end, (end[0] + head * math.cos(a), end[1] + head * math.sin(a))],
            fill=color,
            width=width,
        )


def node(
    draw: Surface,
    box: tuple[float, float, float, float],
    label: str,
    accent: tuple[int, int, int] = CYAN,
    label_font: ImageFont.FreeTypeFont | None = None,
    sub: str | None = None,
    sub_font: ImageFont.FreeTypeFont | None = None,
    fill: tuple[int, int, int, int] = PANEL,
    radius: int = 14,
) -> None:
    """Draw a labelled node box with an accent stripe."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=(*accent, 235), width=2)
    draw.rounded_rectangle((x0 + 1, y0 + 9, x0 + 5, y1 - 9), radius=2, fill=(*accent, 255))
    if label_font is None:
        label_font = font(FONT_MEDIUM, 21)
    cx = (x0 + x1) / 2
    if sub and sub_font:
        draw.text((cx, (y0 + y1) / 2 - 15), label, font=label_font, fill=TEXT, anchor="mm")
        draw.text((cx, (y0 + y1) / 2 + 14), sub, font=sub_font, fill=MUTED, anchor="mm")
    else:
        draw.text((cx, (y0 + y1) / 2), label, font=label_font, fill=TEXT, anchor="mm")


def pill(
    draw: Surface,
    x: float,
    y: float,
    text: str,
    f: ImageFont.FreeTypeFont,
    accent: tuple[int, int, int] = CYAN,
    pad: float = 18,
    height: float = 44,
) -> float:
    """Draw a rounded tag and return its total width."""
    width = draw.textlength(text, font=f) + pad * 2
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=height / 2,
        fill=(*accent, 26),
        outline=(*accent, 190),
        width=2,
    )
    draw.text((x + width / 2, y + height / 2), text, font=f, fill=TEXT, anchor="mm")
    return width


# --------------------------------------------------------------------------- #
# Asset 1 — banner
# --------------------------------------------------------------------------- #


def build_banner() -> Image.Image:
    """Hero banner for the top of the README."""
    width, height = 1600, 520
    base = canvas(width, height)
    base = glow(base, (1270, 70), 380, VIOLET, alpha=115, blur=130)
    base = glow(base, (250, 520), 340, CYAN, alpha=95, blur=130)
    base = glow(base, (900, 260), 420, PINK, alpha=34, blur=150)
    base = grid(base)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = draw_on(overlay)
    # Decorative node graph on the right.
    graph_center = (1290, 300)
    for i, vertex in enumerate(hexagon(*graph_center, 132)):
        od.line([graph_center, vertex], fill=(*CYAN, 90), width=2)
        color = (CYAN, VIOLET, BLUE, GREEN, PINK, AMBER)[i]
        od.ellipse(
            [vertex[0] - 17, vertex[1] - 17, vertex[0] + 17, vertex[1] + 17], fill=(*color, 70)
        )
        od.ellipse(
            [vertex[0] - 10, vertex[1] - 10, vertex[0] + 10, vertex[1] + 10], fill=(*color, 255)
        )
    od.ellipse(
        [graph_center[0] - 40, graph_center[1] - 40, graph_center[0] + 40, graph_center[1] + 40],
        fill=(*VIOLET, 120),
    )
    od.ellipse(
        [graph_center[0] - 24, graph_center[1] - 24, graph_center[0] + 24, graph_center[1] + 24],
        fill=(*TEXT, 235),
    )
    base = Image.alpha_composite(base, overlay)

    d = draw_on(base)
    f_eyebrow = font(FONT_MONO, 22)
    f_title = font(FONT_DISPLAY, 112)
    f_sub = font(FONT_MEDIUM, 35)
    f_tag = font(FONT_REGULAR, 24)
    f_pill = font(FONT_MEDIUM, 22)

    # Eyebrow: hexagon mark + label.
    d.polygon(hexagon(96, 88, 20), outline=(*CYAN, 255), width=3)
    d.polygon(hexagon(96, 88, 8), fill=(*CYAN, 255))
    tracked(d, (130, 74), "ORCHESTRATION PLATFORM", f_eyebrow, MUTED, spacing=4.0)

    tracked(d, (78, 132), "LANGGRAPH", f_title, TEXT, spacing=9.0)
    d.rounded_rectangle((82, 276, 300, 284), radius=4, fill=(*CYAN, 255))
    d.rounded_rectangle((300, 276, 420, 284), radius=4, fill=(*VIOLET, 255))

    tracked(d, (80, 302), "MULTI-AGENT SYSTEM", f_sub, (147, 164, 204), spacing=13.0)
    d.text(
        (82, 356),
        "Typed state  ·  Durable checkpointing  ·  Human-in-the-loop approval",
        font=f_tag,
        fill=(110, 127, 168),
    )
    d.text(
        (82, 386),
        "Provider-independent LLMs  ·  Least-privilege tooling",
        font=f_tag,
        fill=(110, 127, 168),
    )

    x = 82.0
    for label, accent in (
        ("LangGraph", CYAN),
        ("FastAPI", GREEN),
        ("PostgreSQL", BLUE),
        ("Redis", RED),
        ("Docker", VIOLET),
        ("Pydantic v2", AMBER),
    ):
        x += pill(d, x, 424, label, f_pill, accent, pad=15) + 11

    return base


# --------------------------------------------------------------------------- #
# Asset 2 — logo
# --------------------------------------------------------------------------- #


def build_logo() -> Image.Image:
    """Square brand mark."""
    size = 512
    pixels = size * SCALE
    base = Image.new("RGBA", (pixels, pixels), (0, 0, 0, 0))
    tile = canvas(size, size, (14, 18, 40), (4, 6, 16))
    mask = Image.new("L", (pixels, pixels), 0)
    draw_on(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=104, fill=255)
    base.paste(tile, (0, 0), mask)
    base = glow(base, (256, 180), 200, VIOLET, alpha=120, blur=110)
    base = glow(base, (150, 380), 170, CYAN, alpha=90, blur=110)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = draw_on(overlay)
    center = (256.0, 256.0)
    vertices = hexagon(*center, 142)
    for vertex in vertices:
        od.line([center, vertex], fill=(*CYAN, 120), width=3)
    od.ellipse(
        [center[0] - 52, center[1] - 52, center[0] + 52, center[1] + 52], fill=(*VIOLET, 150)
    )
    base = Image.alpha_composite(base, overlay)

    d = draw_on(base)
    d.polygon(vertices, outline=(*CYAN, 255), width=4)
    for vertex in vertices:
        d.ellipse(
            [vertex[0] - 15, vertex[1] - 15, vertex[0] + 15, vertex[1] + 15], fill=(*CYAN, 255)
        )
    d.ellipse([center[0] - 30, center[1] - 30, center[0] + 30, center[1] + 30], fill=(*TEXT, 255))
    d.ellipse([center[0] - 14, center[1] - 14, center[0] + 14, center[1] + 14], fill=(*VIOLET, 255))
    return base


# --------------------------------------------------------------------------- #
# Asset 3 — layered architecture
# --------------------------------------------------------------------------- #


def build_architecture() -> Image.Image:
    """Layered architecture view: edge, orchestration, agents, capability, infrastructure."""
    width, height = 1760, 1180
    base = canvas(width, height)
    base = glow(base, (300, 120), 380, VIOLET, alpha=72, blur=140)
    base = glow(base, (1500, 1040), 400, CYAN, alpha=64, blur=150)
    base = grid(base)

    d = draw_on(base)
    f_title = font(FONT_DISPLAY, 36)
    f_head = font(FONT_DISPLAY, 26)
    f_meta = font(FONT_REGULAR, 22)
    f_band = font(FONT_MONO_BOLD, 23)

    d.text((60, 34), "SYSTEM ARCHITECTURE", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
        "Five strictly separated layers. Each layer depends only on the abstractions below it.",
        font=f_meta,
        fill=MUTED,
    )

    bands: list[tuple[str, tuple[int, int, int], list[tuple[str, str]]]] = [
        (
            "EDGE / API",
            GREEN,
            [
                ("FastAPI App", "ASGI"),
                ("Request Validation", "Pydantic"),
                ("Middleware", "auth · rate limit"),
                ("Route Handlers", "chat · tasks · approvals"),
            ],
        ),
        (
            "ORCHESTRATION",
            CYAN,
            [
                ("Task Manager", "lifecycle"),
                ("LangGraph Runtime", "typed state"),
                ("Intent Router", "structured output"),
                ("Planner", "plan + validate"),
                ("Critic", "verify outputs"),
                ("Synthesizer", "final answer"),
            ],
        ),
        (
            "AGENTS",
            BLUE,
            [
                ("Research", "evidence"),
                ("Coding", "generate · debug"),
                ("Analysis", "compute"),
                ("Executor", "approved actions"),
            ],
        ),
        (
            "CAPABILITY",
            VIOLET,
            [
                ("Tool Registry", "least privilege"),
                ("Memory Manager", "4 tiers"),
                ("Checkpoint Store", "resume"),
                ("LLM Provider", "vendor-neutral"),
                ("Embedding Provider", "vendor-neutral"),
            ],
        ),
        (
            "INFRASTRUCTURE",
            AMBER,
            [
                ("PostgreSQL", "durable"),
                ("Redis", "cache · limits"),
                ("Object / Event Log", "audit trail"),
            ],
        ),
    ]

    y = 118.0
    band_height = 186.0
    gap = 22.0
    left, right = 60.0, width - 60.0

    for title, accent, items in bands:
        box = (left, y, right, y + band_height)
        d.rounded_rectangle(box, radius=20, fill=PANEL_SOFT, outline=(*BORDER, 255), width=2)
        d.rounded_rectangle((left, y, left + 6, y + band_height), radius=3, fill=(*accent, 255))
        tracked(d, (left + 30, y + 16), title, f_band, accent, spacing=2.4)

        inner_left, inner_right = left + 30, right - 26
        inner_top, inner_bottom = y + 56, y + band_height - 24
        count = len(items)
        spacing = 20
        box_w = (inner_right - inner_left - spacing * (count - 1)) / count
        for index, (name, meta) in enumerate(items):
            bx = inner_left + index * (box_w + spacing)
            node(
                d,
                (bx, inner_top, bx + box_w, inner_bottom),
                name,
                accent,
                f_head,
                meta,
                f_meta,
                radius=12,
            )

        y += band_height + gap

    # Vertical connectors down the centre of the stack.
    mid = (left + right) / 2
    for index in range(len(bands) - 1):
        y_start = 118 + index * (band_height + gap) + band_height
        arrow(d, (mid, y_start + 2), (mid, y_start + gap - 2), (90, 106, 150), width=3, head=10)

    return base


# --------------------------------------------------------------------------- #
# Asset 4 — graph execution flow
# --------------------------------------------------------------------------- #


def build_graph_flow() -> Image.Image:
    """LangGraph node graph with conditional branches and a retry loop.

    Every box here is a node that exists in ``app/graph/builder.py``. An earlier
    version of this diagram showed a ``dispatch_tasks`` node that was never
    registered, which is worse than an incomplete diagram: it documents a
    control flow that does not exist. The terminal nodes that the main chain
    omits are named in the caption instead, so nothing is implied to be absent.
    """
    width, height = 1820, 1500
    base = canvas(width, height)
    base = glow(base, (880, 60), 420, VIOLET, alpha=62, blur=150)
    base = glow(base, (1560, 900), 380, AMBER, alpha=52, blur=150)
    base = glow(base, (200, 1300), 340, GREEN, alpha=48, blur=150)
    base = grid(base)

    d = draw_on(base)
    f_title = font(FONT_DISPLAY, 36)
    f_meta = font(FONT_REGULAR, 22)
    f_node = font(FONT_MEDIUM, 26)
    f_kind = font(FONT_MONO, 22)
    f_edge = font(FONT_MONO_BOLD, 22)

    d.text((60, 30), "GRAPH EXECUTION FLOW", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
        "Conditional edges decide every branch. Every cycle is bounded by the "
        "configured iteration and retry limits.",
        font=f_meta,
        fill=MUTED,
    )
    d.text(
        (62, 106),
        "Terminal branches not drawn: cancel_task, fail_task, and record_memory, "
        "which every path passes through before finalize.",
        font=f_meta,
        fill=DIM,
    )

    main_x = 700.0
    branch_x = 1360.0
    node_w, node_h = 330.0, 74.0
    step = 96.0
    top = 168.0

    def main_box(row: int) -> tuple[float, float, float, float]:
        y = top + row * step
        return (main_x - node_w / 2, y, main_x + node_w / 2, y + node_h)

    def branch_box(row: int) -> tuple[float, float, float, float]:
        y = top + row * step
        return (branch_x - node_w / 2, y, branch_x + node_w / 2, y + node_h)

    def connector(row: int) -> None:
        y_from = top + row * step + node_h
        y_to = top + (row + 1) * step
        arrow(d, (main_x, y_from + 3), (main_x, y_to - 3), (86, 102, 148), width=3, head=12)

    # Row numbers below are the order the nodes actually run in, so every
    # straight connector is between adjacent rows and no gap needs explaining.
    # Subtitles are deliberately terse: at this width a longer caption overflows
    # the node, and the prose around the diagram carries the detail.
    node(d, main_box(0), "START", GREEN, f_node, "entry point", f_kind)
    node(d, main_box(1), "validate_input", CYAN, f_node, "reject bad input", f_kind)
    node(d, main_box(2), "recall_memory", VIOLET, f_node, "relevance-gated", f_kind)
    node(d, main_box(3), "route_request", AMBER, f_node, "intent · risk", f_kind)
    node(d, main_box(4), "planner", BLUE, f_node, "structured plan", f_kind)
    node(d, main_box(5), "validate_plan", CYAN, f_node, "schema + deps", f_kind)
    node(d, main_box(6), "agent_execution", BLUE, f_node, "bounded fan-out", f_kind)
    node(d, main_box(7), "aggregate_results", CYAN, f_node, "merge outputs", f_kind)
    node(d, main_box(8), "critic", AMBER, f_node, "PASS / FAIL", f_kind)
    node(d, main_box(9), "synthesizer", BLUE, f_node, "final answer", f_kind)
    node(d, main_box(10), "risk_check", AMBER, f_node, "approval needed?", f_kind)
    node(d, main_box(11), "finalize", GREEN, f_node, "persist · respond", f_kind)
    node(d, main_box(12), "END", GREEN, f_node, "terminal", f_kind)

    for row in range(12):
        connector(row)

    def edge_label(row: int, text: str, colour: tuple[int, int, int]) -> None:
        """Label the connector leaving ``row``, beside the line rather than on it."""
        midpoint = (main_box(row)[3] + main_box(row + 1)[1]) / 2
        d.text((main_x + 18, midpoint - 12), text, font=f_edge, fill=colour)

    edge_label(3, "COMPLEX", MUTED)
    edge_label(8, "PASS", GREEN)
    edge_label(10, "NO", MUTED)

    # Route branch: a simple request short-circuits down a dedicated right-hand
    # channel, so it never draws across the approval branch.
    node(d, branch_box(3), "direct_response", GREEN, f_node, "no planning", f_kind)
    arrow(
        d,
        (main_box(3)[2], main_box(3)[1] + node_h / 2),
        (branch_box(3)[0] - 3, branch_box(3)[1] + node_h / 2),
        GREEN,
        width=3,
    )
    d.text(
        (main_box(3)[2] + 14, main_box(3)[1] + node_h / 2 - 34), "SIMPLE", font=f_edge, fill=GREEN
    )
    simple_channel = 1750.0
    end_y = main_box(12)[1] + node_h / 2
    simple_y = branch_box(3)[1] + node_h / 2
    d.line([(branch_box(3)[2], simple_y), (simple_channel, simple_y)], fill=GREEN, width=3)
    d.line([(simple_channel, simple_y), (simple_channel, end_y)], fill=GREEN, width=3)
    arrow(d, (simple_channel, end_y), (main_box(12)[2] + 4, end_y), GREEN, width=3)
    d.text((simple_channel - 232, simple_y - 34), "respond directly", font=f_edge, fill=MUTED)

    # Critic failure path loops back to agent_execution, which is where the work
    # would be redone.
    node(d, branch_box(8), "retry_or_replan", RED, f_node, "if budget remains", f_kind)
    arrow(
        d,
        (main_box(8)[2], main_box(8)[1] + node_h / 2),
        (branch_box(8)[0] - 3, branch_box(8)[1] + node_h / 2),
        RED,
        width=3,
    )
    d.text((main_box(8)[2] + 14, main_box(8)[1] + node_h / 2 - 34), "FAIL", font=f_edge, fill=RED)
    loop_x = 1630.0
    loop_y = main_box(6)[1] + node_h / 2
    failure_y = branch_box(8)[1] + node_h / 2
    d.line([(branch_box(8)[2], failure_y), (loop_x, failure_y)], fill=RED, width=3)
    d.line([(loop_x, failure_y), (loop_x, loop_y)], fill=RED, width=3)
    arrow(d, (loop_x, loop_y), (main_box(6)[2] + 4, loop_y), RED, width=3)
    d.text((loop_x - 322, loop_y - 34), "REPLAN", font=f_edge, fill=RED)

    # Approval branch. Reaching execute_approved_action requires the run to have
    # been resumed with a decision, so it sits directly beneath the interrupt.
    node(d, branch_box(10), "human_approval", VIOLET, f_node, "graph interrupts", f_kind)
    node(d, branch_box(11), "execute / cancel", PINK, f_node, "on resume", f_kind)
    arrow(
        d,
        (main_box(10)[2], main_box(10)[1] + node_h / 2),
        (branch_box(10)[0] - 3, branch_box(10)[1] + node_h / 2),
        VIOLET,
        width=3,
    )
    d.text(
        (main_box(10)[2] + 14, main_box(10)[1] + node_h / 2 - 34), "YES", font=f_edge, fill=VIOLET
    )
    arrow(d, (branch_x, branch_box(10)[3] + 3), (branch_x, branch_box(11)[1] - 3), VIOLET, width=3)
    arrow(
        d,
        (branch_box(11)[0] - 3, branch_box(11)[1] + node_h / 2),
        (main_box(11)[2] + 4, main_box(11)[1] + node_h / 2),
        PINK,
        width=3,
    )

    return base


# --------------------------------------------------------------------------- #
# Asset 5 — tool security pipeline
# --------------------------------------------------------------------------- #


def build_tool_security() -> Image.Image:
    """The mandatory pipeline every tool call passes through, plus the risk ladder."""
    width, height = 1760, 780
    base = canvas(width, height)
    base = glow(base, (880, 80), 420, VIOLET, alpha=66, blur=140)
    base = glow(base, (1420, 700), 380, RED, alpha=58, blur=150)
    base = grid(base)

    d = draw_on(base)
    f_title = font(FONT_DISPLAY, 36)
    f_meta = font(FONT_REGULAR, 22)
    f_stage = font(FONT_MEDIUM, 24)
    f_step = font(FONT_MONO_BOLD, 22)
    f_risk = font(FONT_DISPLAY, 30)
    f_risk_meta = font(FONT_REGULAR, 21)

    d.text((60, 30), "TOOL CALL SECURITY PIPELINE", font=f_title, fill=TEXT)
    d.text(
        (62, 80),
        "No tool executes outside this pipeline. Permission, risk, and approval "
        "checks all precede execution.",
        font=f_meta,
        fill=MUTED,
    )

    stages = [
        ("ToolRequest", CYAN),
        ("Input Validation", CYAN),
        ("Permission Check", BLUE),
        ("Risk Classification", AMBER),
        ("Approval Check", VIOLET),
        ("Execution", GREEN),
        ("Result Validation", CYAN),
        ("Audit Event", BLUE),
    ]

    # Eight stages in one row left roughly 190 pixels each, which forced labels
    # small enough to be unreadable once the README scaled the image down.
    # Two rows of four give each label twice the width, so a legible size fits.
    left, right = 60.0, width - 60.0
    per_row = 4
    gap = 22.0
    box_w = (right - left - gap * (per_row - 1)) / per_row
    row_height = 84.0
    row_ys = (140.0, 258.0)

    for index, (label, accent) in enumerate(stages):
        row, column = divmod(index, per_row)
        # Serpentine order: the flow reads left to right, then down and back, so
        # a single connector joins the two rows instead of a long diagonal.
        if row % 2 == 1:
            column = per_row - 1 - column
        x = left + column * (box_w + gap)
        y = row_ys[row]
        node(d, (x, y, x + box_w, y + row_height), label, accent, f_stage, radius=12)
        d.text((x + 14, y - 30), f"0{index + 1}", font=f_step, fill=DIM)

        following = index + 1
        if following < len(stages) and following // per_row == row:
            direction = 1 if row % 2 == 0 else -1
            edge_x = x + box_w + 3 if direction == 1 else x - 3
            target_x = edge_x + direction * (gap - 6)
            arrow(
                d,
                (edge_x, y + row_height / 2),
                (target_x, y + row_height / 2),
                (86, 102, 148),
                width=3,
                head=11,
            )

    # The first row ends and the second begins at the same column, because the
    # second row runs right to left. One short vertical arrow joins them.
    seam_x = right - box_w / 2
    arrow(
        d,
        (seam_x, row_ys[0] + row_height + 4),
        (seam_x, row_ys[1] - 4),
        (86, 102, 148),
        width=3,
        head=11,
    )

    d.text((60, 366), "RISK CLASSIFICATION", font=f_step, fill=DIM)

    risks = [
        ("LOW", GREEN, "Read file, search web", "auto-execute"),
        ("MEDIUM", AMBER, "Write file, create issue", "audited"),
        ("HIGH", (251, 146, 60), "Execute code, write to DB", "requires approval"),
        ("CRITICAL", RED, "Delete data, drop table", "requires approval"),
    ]
    r_left, r_right = 60.0, width - 60.0
    r_gap = 20.0
    r_w = (r_right - r_left - r_gap * 3) / 4
    r_top, r_bottom = 404.0, 726.0

    for index, (level, accent, example, policy) in enumerate(risks):
        x = r_left + index * (r_w + r_gap)
        d.rounded_rectangle(
            (x, r_top, x + r_w, r_bottom),
            radius=18,
            fill=PANEL_SOFT,
            outline=(*accent, 210),
            width=2,
        )
        d.rounded_rectangle((x, r_top, x + r_w, r_top + 8), radius=4, fill=(*accent, 255))
        d.text((x + 26, r_top + 38), level, font=f_risk, fill=accent)
        d.text((x + 26, r_top + 96), "example", font=f_step, fill=DIM)
        d.text((x + 26, r_top + 126), example, font=f_risk_meta, fill=TEXT)
        d.text((x + 26, r_top + 196), "policy", font=f_step, fill=DIM)
        d.text((x + 26, r_top + 226), policy, font=f_risk_meta, fill=accent)

    return base


# --------------------------------------------------------------------------- #
# Asset 6 — memory architecture
# --------------------------------------------------------------------------- #


def build_memory() -> Image.Image:
    """Four memory tiers behind a single manager."""
    width, height = 1760, 620
    base = canvas(width, height)
    base = glow(base, (300, 90), 380, CYAN, alpha=64, blur=140)
    base = glow(base, (1450, 560), 380, VIOLET, alpha=60, blur=150)
    base = grid(base)

    d = draw_on(base)
    f_title = font(FONT_DISPLAY, 36)
    f_meta = font(FONT_REGULAR, 22)
    f_node = font(FONT_MEDIUM, 26)
    f_kind = font(FONT_MONO, 22)
    f_step = font(FONT_MONO_BOLD, 22)
    f_body = font(FONT_REGULAR, 22)

    d.text((60, 34), "MEMORY ARCHITECTURE", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
        "Four tiers, one manager. Relevance is evaluated before anything is "
        "promoted to durable storage.",
        font=f_meta,
        fill=MUTED,
    )

    node(
        d,
        (470, 122, 1290, 204),
        "MemoryManager",
        VIOLET,
        font(FONT_DISPLAY, 32),
        "retrieve · store · update · summarize · delete",
        f_kind,
        radius=16,
    )

    tiers = [
        ("SHORT-TERM", CYAN, "Current conversation context", "in-process buffer"),
        ("WORKING", BLUE, "State of the task in flight", "graph state"),
        ("LONG-TERM SEMANTIC", VIOLET, "Durable facts worth keeping", "PostgreSQL + vectors"),
        ("EXECUTION", GREEN, "What previous runs did", "execution records"),
    ]

    left, right = 60.0, width - 60.0
    gap = 22.0
    box_w = (right - left - gap * 3) / 4
    top, bottom = 280.0, 556.0

    for index, (name, accent, desc, store) in enumerate(tiers):
        x = left + index * (box_w + gap)
        arrow(d, (x + box_w / 2, 208), (x + box_w / 2, top - 6), accent, width=3, head=11)
        d.rounded_rectangle(
            (x, top, x + box_w, bottom), radius=18, fill=PANEL_SOFT, outline=(*accent, 205), width=2
        )
        d.rounded_rectangle((x, top, x + box_w, top + 8), radius=4, fill=(*accent, 255))
        tracked(d, (x + 26, top + 34), name, f_step, accent, spacing=1.6)
        for offset, line in enumerate(wrap_text(d, desc, f_node, box_w - 52, max_lines=2)):
            d.text((x + 26, top + 80 + offset * 33), line, font=f_node, fill=TEXT)
        d.text((x + 26, top + 166), "storage", font=f_step, fill=DIM)
        d.text((x + 26, top + 196), store, font=f_body, fill=MUTED)

    return base


# --------------------------------------------------------------------------- #
# Asset 7 — roadmap
# --------------------------------------------------------------------------- #


def build_roadmap() -> Image.Image:
    """Phase timeline with the current position marked."""
    width, height = 1760, 640
    base = canvas(width, height)
    base = glow(base, (240, 520), 360, GREEN, alpha=66, blur=140)
    base = glow(base, (1500, 120), 400, VIOLET, alpha=62, blur=150)
    base = grid(base)

    d = draw_on(base)
    f_title = font(FONT_DISPLAY, 36)
    f_meta = font(FONT_REGULAR, 22)
    f_disp = font(FONT_DISPLAY, 26)
    f_step = font(FONT_MONO_BOLD, 22)
    f_body = font(FONT_REGULAR, 22)

    d.text((60, 34), "BUILD ROADMAP", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
        "Phases are sequential. Each one passes format, lint, type check, and "
        "tests before the next begins.",
        font=f_meta,
        fill=MUTED,
    )

    track_y = 336.0
    # Inset by the half-width of a card so that the first and last cards sit
    # inside the canvas. The original inset was 110 pixels against a 148-pixel
    # half-width, which clipped both end cards and their text.
    left, right = 200.0, width - 200.0
    d.line([(left, track_y), (right, track_y)], fill=(58, 72, 116), width=4)

    milestones = [
        ("0-1", "FOUNDATION", "Environment, tooling, health endpoint", GREEN, "complete"),
        ("2-6", "SERVICES", "Config, database, Redis, LLM + embedding abstractions", CYAN, "next"),
        (
            "7-24",
            "CORE",
            "State, tools, agents, graph, parallelism, memory, approval",
            BLUE,
            "planned",
        ),
        (
            "25-33",
            "SECURITY",
            "Authorization, rate limiting, SSRF, injection defenses",
            AMBER,
            "planned",
        ),
        ("34-45", "HARDENING", "Testing, evaluation, Docker, CI/CD, reviews", VIOLET, "planned"),
    ]

    count = len(milestones)
    spacing = (right - left) / (count - 1)

    for index, (phase, name, detail, accent, status) in enumerate(milestones):
        cx = left + index * spacing
        d.ellipse([cx - 19, track_y - 19, cx + 19, track_y + 19], fill=(*accent, 60))
        d.ellipse([cx - 12, track_y - 12, cx + 12, track_y + 12], fill=(*accent, 255))
        if status == "complete":
            d.ellipse([cx - 5, track_y - 5, cx + 5, track_y + 5], fill=(*BG_BOTTOM, 255))

        # Progress fill up to the last completed milestone.
        if index == 0:
            d.line([(cx, track_y), (cx + spacing, track_y)], fill=(*GREEN, 255), width=4)

        # Cards alternate above and below the track, and both must stay clear of
        # it: the connector arrows would otherwise be drawn through a card.
        box_top = 140.0 if index % 2 == 0 else 380.0
        box = (cx - 148, box_top, cx + 148, box_top + 166)
        d.rounded_rectangle(box, radius=16, fill=PANEL_SOFT, outline=(*accent, 200), width=2)
        tracked(d, (cx - 126, box_top + 18), phase, f_step, accent, spacing=1.6)
        d.text((cx - 126, box_top + 50), name, font=f_disp, fill=TEXT)
        for offset, line in enumerate(wrap_text(d, detail, f_body, 262, max_lines=2)):
            d.text((cx - 126, box_top + 90 + offset * 25), line, font=f_body, fill=MUTED)
        d.text((cx - 126, box_top + 140), status.upper(), font=f_step, fill=accent)
        target = box[3] if index % 2 == 0 else box[1]
        arrow(
            d,
            (cx, track_y + (22 if index % 2 == 0 else -22)),
            (cx, target + (-4 if index % 2 == 0 else 4)),
            accent,
            width=2,
            head=9,
        )

    return base


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

#: Each builder paired with its canvas size in design units, so the audit can
#: check that nothing was drawn past the edge.
BUILDERS: dict[str, tuple[Callable[[], Image.Image], tuple[int, int]]] = {
    "banner.png": (build_banner, (1600, 520)),
    "logo.png": (build_logo, (512, 512)),
    "architecture.png": (build_architecture, (1760, 1180)),
    "graph-flow.png": (build_graph_flow, (1820, 1500)),
    "tool-security.png": (build_tool_security, (1760, 780)),
    "memory.png": (build_memory, (1760, 620)),
    "roadmap.png": (build_roadmap, (1760, 640)),
}


def main() -> int:
    """Render every asset, audit its layout, and report what was produced."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"writing to {ASSETS}")

    problems = 0
    for name, (builder, design) in BUILDERS.items():
        _RECORDED_TEXT.clear()
        image = builder()

        # The registered size drives the audit, so a stale entry would silently
        # disable the "nothing escapes the canvas" check along one axis.
        if image.size != (design[0] * SCALE, design[1] * SCALE):
            raise RuntimeError(
                f"{name}: declared {design} but rendered "
                f"{image.size[0] // SCALE}x{image.size[1] // SCALE}; "
                "update the BUILDERS entry"
            )

        for problem in audit_layout(*design):
            problems += 1
            print(f"  ! {name}: {problem}")

        target = ASSETS / name
        image.save(target, optimize=True, dpi=(DPI, DPI))

        # Guard against silently emitting a blank canvas. A histogram is used
        # rather than getextrema() because the latter is typed as a union that
        # depends on the image mode.
        histogram = image.convert("L").histogram()
        low = next(index for index, count in enumerate(histogram) if count)
        high = 255 - next(index for index, count in enumerate(reversed(histogram)) if count)
        if high - low < 20:
            raise RuntimeError(f"{name} looks blank (luminance range {low}-{high})")

        size_kb = target.stat().st_size / 1024
        print(f"  {name:<20} {image.size[0]}x{image.size[1]}  {size_kb:6.1f} KB")

    print(f"done: {len(BUILDERS)} assets, {problems} layout problem(s)")
    # A non-zero exit is the point of the audit: a diagram that silently pushed
    # a label off its own canvas is a documentation defect, and one this build
    # can detect for free.
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
