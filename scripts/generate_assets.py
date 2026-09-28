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
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- #
# Paths and fonts
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"
FONT_ROOT = Path("/usr/share/fonts/truetype")

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


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at the given pixel size."""
    return ImageFont.truetype(str(path), size)


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


def canvas(
    width: int,
    height: int,
    top: tuple[int, int, int] = BG_TOP,
    bottom: tuple[int, int, int] = BG_BOTTOM,
) -> Image.Image:
    """Create the base RGBA canvas for an asset."""
    return vertical_gradient(width, height, top, bottom).convert("RGBA")


def glow(
    base: Image.Image,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: int = 90,
    blur: int = 110,
) -> Image.Image:
    """Composite a soft radial bloom onto the canvas."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(*color, alpha))
    return Image.alpha_composite(base, layer.filter(ImageFilter.GaussianBlur(blur)))


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
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(*color, alpha), width=1)
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill=(*color, alpha), width=1)
    return Image.alpha_composite(base, layer)


def text_width(
    draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, spacing: float = 0.0
) -> float:
    """Measure a string including letter spacing."""
    return sum(draw.textlength(c, font=f) for c in text) + spacing * max(len(text) - 1, 0)


def tracked(
    draw: ImageDraw.ImageDraw,
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
    draw: ImageDraw.ImageDraw,
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
    draw: ImageDraw.ImageDraw,
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
    draw: ImageDraw.ImageDraw,
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
    od = ImageDraw.Draw(overlay)
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

    d = ImageDraw.Draw(base)
    f_eyebrow = font(FONT_MONO, 18)
    f_title = font(FONT_DISPLAY, 108)
    f_sub = font(FONT_MEDIUM, 31)
    f_tag = font(FONT_REGULAR, 21)
    f_pill = font(FONT_MEDIUM, 19)

    # Eyebrow: hexagon mark + label.
    d.polygon(hexagon(96, 88, 20), outline=(*CYAN, 255), width=3)
    d.polygon(hexagon(96, 88, 8), fill=(*CYAN, 255))
    tracked(d, (130, 76), "ORCHESTRATION PLATFORM", f_eyebrow, MUTED, spacing=4.0)

    tracked(d, (78, 138), "LANGGRAPH", f_title, TEXT, spacing=9.0)
    d.rounded_rectangle((82, 272, 300, 279), radius=4, fill=(*CYAN, 255))
    d.rounded_rectangle((300, 272, 420, 279), radius=4, fill=(*VIOLET, 255))

    tracked(d, (80, 300), "MULTI-AGENT SYSTEM", f_sub, (147, 164, 204), spacing=13.0)
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
        x += pill(d, x, 430, label, f_pill, accent) + 12

    return base


# --------------------------------------------------------------------------- #
# Asset 2 — logo
# --------------------------------------------------------------------------- #


def build_logo() -> Image.Image:
    """Square brand mark."""
    size = 512
    base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile = canvas(size, size, (14, 18, 40), (4, 6, 16))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=104, fill=255)
    base.paste(tile, (0, 0), mask)
    base = glow(base, (256, 180), 200, VIOLET, alpha=120, blur=110)
    base = glow(base, (150, 380), 170, CYAN, alpha=90, blur=110)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    center = (256.0, 256.0)
    vertices = hexagon(*center, 142)
    for vertex in vertices:
        od.line([center, vertex], fill=(*CYAN, 120), width=3)
    od.ellipse(
        [center[0] - 52, center[1] - 52, center[0] + 52, center[1] + 52], fill=(*VIOLET, 150)
    )
    base = Image.alpha_composite(base, overlay)

    d = ImageDraw.Draw(base)
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

    d = ImageDraw.Draw(base)
    f_title = font(FONT_DISPLAY, 30)
    f_head = font(FONT_DISPLAY, 22)
    f_meta = font(FONT_REGULAR, 17)
    f_band = font(FONT_MONO_BOLD, 19)

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
                ("Document", "extract"),
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
    """LangGraph node graph with conditional branches and a retry loop."""
    width, height = 1820, 1650
    base = canvas(width, height)
    base = glow(base, (880, 60), 420, VIOLET, alpha=62, blur=150)
    base = glow(base, (1560, 900), 380, AMBER, alpha=52, blur=150)
    base = glow(base, (200, 1300), 340, GREEN, alpha=48, blur=150)
    base = grid(base)

    d = ImageDraw.Draw(base)
    f_title = font(FONT_DISPLAY, 30)
    f_meta = font(FONT_REGULAR, 17)
    f_node = font(FONT_MEDIUM, 22)
    f_kind = font(FONT_MONO, 16)
    f_edge = font(FONT_MONO_BOLD, 16)

    d.text((60, 34), "GRAPH EXECUTION FLOW", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
        "Conditional edges decide every branch. Every cycle is bounded by the "
        "configured iteration and retry limits.",
        font=f_meta,
        fill=MUTED,
    )

    main_x = 700.0
    branch_x = 1360.0
    node_w, node_h = 330.0, 74.0
    step = 92.0
    top = 128.0

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

    # Main chain.
    node(d, main_box(0), "START", GREEN, f_node, "entry point", f_kind)
    node(d, main_box(1), "validate_input", CYAN, f_node, "reject malformed requests", f_kind)
    node(d, main_box(2), "route_request", AMBER, f_node, "intent · complexity · risk", f_kind)
    node(d, main_box(4), "planner", BLUE, f_node, "structured plan", f_kind)
    node(d, main_box(5), "validate_plan", CYAN, f_node, "schema + dependency check", f_kind)
    node(d, main_box(6), "dispatch_tasks", VIOLET, f_node, "bounded parallelism", f_kind)
    node(d, main_box(7), "agent_execution", BLUE, f_node, "research · coding · analysis", f_kind)
    node(d, main_box(8), "aggregate_results", CYAN, f_node, "merge agent outputs", f_kind)
    node(d, main_box(9), "critic", AMBER, f_node, "PASS / FAIL", f_kind)
    node(d, main_box(11), "synthesizer", BLUE, f_node, "verified final answer", f_kind)
    node(d, main_box(12), "risk_check", AMBER, f_node, "approval required?", f_kind)
    node(d, main_box(14), "finalize", GREEN, f_node, "persist · respond", f_kind)
    node(d, main_box(15), "END", GREEN, f_node, "terminal", f_kind)

    for row in (0, 1, 4, 5, 6, 7, 8, 11, 14):
        connector(row)

    # route_request -> planner, skipping the unused row 3.
    arrow(
        d,
        (main_x, main_box(2)[3] + 3),
        (main_x, main_box(4)[1] - 3),
        (86, 102, 148),
        width=3,
        head=12,
    )
    d.text(
        (main_x + 18, (main_box(2)[3] + main_box(4)[1]) / 2 - 9),
        "COMPLEX -> plan",
        font=f_edge,
        fill=MUTED,
    )

    # critic -> synthesizer, the passing path, skipping the unused row 10.
    arrow(d, (main_x, main_box(9)[3] + 3), (main_x, main_box(11)[1] - 3), GREEN, width=3, head=12)
    d.text(
        (main_x + 18, (main_box(9)[3] + main_box(11)[1]) / 2 - 9), "PASS", font=f_edge, fill=GREEN
    )

    # risk_check -> finalize when no approval is required, skipping row 13.
    arrow(
        d,
        (main_x, main_box(12)[3] + 3),
        (main_x, main_box(14)[1] - 3),
        (86, 102, 148),
        width=3,
        head=12,
    )
    d.text(
        (main_x + 18, (main_box(12)[3] + main_box(14)[1]) / 2 - 9), "NO", font=f_edge, fill=MUTED
    )

    # Route branch: a simple request short-circuits down a dedicated right-hand
    # channel, so it never draws across the approval branch.
    node(d, branch_box(2), "direct_response", GREEN, f_node, "no planning needed", f_kind)
    arrow(
        d,
        (main_box(2)[2], main_box(2)[1] + node_h / 2),
        (branch_box(2)[0] - 3, branch_box(2)[1] + node_h / 2),
        GREEN,
        width=3,
    )
    d.text(
        (main_box(2)[2] + 14, main_box(2)[1] + node_h / 2 - 34), "SIMPLE", font=f_edge, fill=GREEN
    )
    simple_channel = 1752.0
    end_y = main_box(15)[1] + node_h / 2
    simple_y = branch_box(2)[1] + node_h / 2
    d.line([(branch_box(2)[2], simple_y), (simple_channel, simple_y)], fill=GREEN, width=3)
    d.line([(simple_channel, simple_y), (simple_channel, end_y)], fill=GREEN, width=3)
    arrow(d, (simple_channel, end_y), (main_box(15)[2] + 4, end_y), GREEN, width=3)
    d.text((simple_channel - 226, simple_y - 30), "respond directly", font=f_edge, fill=MUTED)

    # Critic failure path loops back to dispatch_tasks.
    node(d, branch_box(9), "retry_or_replan", RED, f_node, "if budget remains", f_kind)
    arrow(
        d,
        (main_box(9)[2], main_box(9)[1] + node_h / 2),
        (branch_box(9)[0] - 3, branch_box(9)[1] + node_h / 2),
        RED,
        width=3,
    )
    d.text((main_box(9)[2] + 14, main_box(9)[1] + node_h / 2 - 34), "FAIL", font=f_edge, fill=RED)
    loop_x = 1618.0
    loop_y = top + 6 * step + node_h / 2
    failure_y = branch_box(9)[1] + node_h / 2
    d.line([(branch_box(9)[2], failure_y), (loop_x, failure_y)], fill=RED, width=3)
    d.line([(loop_x, failure_y), (loop_x, loop_y)], fill=RED, width=3)
    arrow(d, (loop_x, loop_y), (main_box(6)[2] + 4, loop_y), RED, width=3)
    d.text((loop_x - 300, loop_y - 30), "REPLAN", font=f_edge, fill=RED)

    # Approval branch.
    node(d, branch_box(12), "human_approval", VIOLET, f_node, "graph interrupts", f_kind)
    node(d, branch_box(13), "execute / cancel", PINK, f_node, "on resume", f_kind)
    arrow(
        d,
        (main_box(12)[2], main_box(12)[1] + node_h / 2),
        (branch_box(12)[0] - 3, branch_box(12)[1] + node_h / 2),
        VIOLET,
        width=3,
    )
    d.text(
        (main_box(12)[2] + 14, main_box(12)[1] + node_h / 2 - 34), "YES", font=f_edge, fill=VIOLET
    )
    arrow(d, (branch_x, branch_box(12)[3] + 3), (branch_x, branch_box(13)[1] - 3), VIOLET, width=3)
    arrow(
        d,
        (branch_box(13)[0] - 3, branch_box(13)[1] + node_h / 2),
        (main_box(14)[2] + 4, main_box(14)[1] + node_h / 2),
        PINK,
        width=3,
    )

    return base


# --------------------------------------------------------------------------- #
# Asset 5 — tool security pipeline
# --------------------------------------------------------------------------- #


def build_tool_security() -> Image.Image:
    """The mandatory pipeline every tool call passes through, plus the risk ladder."""
    width, height = 1760, 700
    base = canvas(width, height)
    base = glow(base, (880, 80), 420, VIOLET, alpha=66, blur=140)
    base = glow(base, (1420, 620), 380, RED, alpha=58, blur=150)
    base = grid(base)

    d = ImageDraw.Draw(base)
    f_title = font(FONT_DISPLAY, 30)
    f_meta = font(FONT_REGULAR, 17)
    f_stage = font(FONT_MEDIUM, 19)
    f_step = font(FONT_MONO_BOLD, 15)
    f_risk = font(FONT_DISPLAY, 24)
    f_risk_meta = font(FONT_REGULAR, 16)

    d.text((60, 34), "TOOL CALL SECURITY PIPELINE", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
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

    left, right = 60.0, width - 60.0
    count = len(stages)
    gap = 18.0
    box_w = (right - left - gap * (count - 1)) / count
    top, bottom = 128.0, 216.0

    for index, (label, accent) in enumerate(stages):
        x = left + index * (box_w + gap)
        node(d, (x, top, x + box_w, bottom), label, accent, f_stage, radius=12)
        d.text((x + 12, top - 26), f"0{index + 1}", font=f_step, fill=DIM)
        if index < count - 1:
            arrow(
                d,
                (x + box_w + 3, (top + bottom) / 2),
                (x + box_w + gap - 3, (top + bottom) / 2),
                (86, 102, 148),
                width=3,
                head=9,
            )

    d.text((60, 268), "RISK CLASSIFICATION", font=f_step, fill=DIM)

    risks = [
        ("LOW", GREEN, "Read file, search web", "auto-execute"),
        ("MEDIUM", AMBER, "Write file, create issue", "audited"),
        ("HIGH", (251, 146, 60), "Execute code, write to DB", "requires approval"),
        ("CRITICAL", RED, "Delete data, drop table", "requires approval"),
    ]
    r_left, r_right = 60.0, width - 60.0
    r_gap = 20.0
    r_w = (r_right - r_left - r_gap * 3) / 4
    r_top, r_bottom = 306.0, 640.0

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
        d.text((x + 26, r_top + 34), level, font=f_risk, fill=accent)
        d.text((x + 26, r_top + 82), "example", font=f_step, fill=DIM)
        d.text((x + 26, r_top + 108), example, font=f_risk_meta, fill=TEXT)
        d.text((x + 26, r_top + 168), "policy", font=f_step, fill=DIM)
        d.text((x + 26, r_top + 194), policy, font=f_risk_meta, fill=accent)

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

    d = ImageDraw.Draw(base)
    f_title = font(FONT_DISPLAY, 30)
    f_meta = font(FONT_REGULAR, 17)
    f_node = font(FONT_MEDIUM, 21)
    f_kind = font(FONT_MONO, 16)
    f_step = font(FONT_MONO_BOLD, 15)
    f_body = font(FONT_REGULAR, 16)

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
        (540, 122, 1220, 196),
        "MemoryManager",
        VIOLET,
        font(FONT_DISPLAY, 26),
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
        arrow(d, (x + box_w / 2, 200), (x + box_w / 2, top - 6), accent, width=3, head=11)
        d.rounded_rectangle(
            (x, top, x + box_w, bottom), radius=18, fill=PANEL_SOFT, outline=(*accent, 205), width=2
        )
        d.rounded_rectangle((x, top, x + box_w, top + 8), radius=4, fill=(*accent, 255))
        tracked(d, (x + 26, top + 32), name, f_step, accent, spacing=1.6)
        d.text((x + 26, top + 74), desc, font=f_node, fill=TEXT)
        d.text((x + 26, top + 116), "storage", font=f_step, fill=DIM)
        d.text((x + 26, top + 142), store, font=f_body, fill=MUTED)

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

    d = ImageDraw.Draw(base)
    f_title = font(FONT_DISPLAY, 30)
    f_meta = font(FONT_REGULAR, 17)
    f_disp = font(FONT_DISPLAY, 22)
    f_step = font(FONT_MONO_BOLD, 15)
    f_body = font(FONT_REGULAR, 16)

    d.text((60, 34), "BUILD ROADMAP", font=f_title, fill=TEXT)
    d.text(
        (62, 76),
        "Phases are sequential. Each one passes format, lint, type check, and "
        "tests before the next begins.",
        font=f_meta,
        fill=MUTED,
    )

    track_y = 320.0
    left, right = 110.0, width - 110.0
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

        box_top = 174.0 if index % 2 == 0 else 380.0
        box = (cx - 148, box_top, cx + 148, box_top + 116)
        d.rounded_rectangle(box, radius=16, fill=PANEL_SOFT, outline=(*accent, 200), width=2)
        tracked(d, (cx - 130, box_top + 16), phase, f_step, accent, spacing=1.6)
        d.text((cx - 130, box_top + 44), name, font=f_disp, fill=TEXT)
        d.text((cx - 130, box_top + 78), detail[:40], font=f_body, fill=MUTED)
        d.text((cx - 130, box_top + 98), status.upper(), font=f_step, fill=accent)
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

BUILDERS = {
    "banner.png": build_banner,
    "logo.png": build_logo,
    "architecture.png": build_architecture,
    "graph-flow.png": build_graph_flow,
    "tool-security.png": build_tool_security,
    "memory.png": build_memory,
    "roadmap.png": build_roadmap,
}


def main() -> int:
    """Render every asset and report what was produced."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"writing to {ASSETS}")

    for name, builder in BUILDERS.items():
        image = builder()
        target = ASSETS / name
        image.save(target, optimize=True)

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

    print(f"done: {len(BUILDERS)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
