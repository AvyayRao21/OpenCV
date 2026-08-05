"""
Step 5 of the parallax window: the illusion itself, driven by the mouse.

This is where the screen stops being a picture and becomes a window. Move the
mouse and the room does not slide — it *opens*. Move left and you see more of
the right-hand wall, exactly as you would leaning across a real window frame.
The near shapes sweep across the far ones. The grid on the back wall swings
while the mouth of the room stays nailed to the glass.

**Nothing rotates.** That is the entire trick, and it is worth being blunt
about because it is so counter-intuitive. The virtual camera never turns to
look at anything; its view direction stays perpendicular to the screen at all
times. What changes is the *shape* of the view volume: the frustum is sheared
so that its four side planes always pass exactly through the four physical
corners of the window. This is Robert Kooima's generalized perspective
projection.

If you were to implement this by orbiting a camera around the scene with a
`lookAt()`, you would get step 4's behaviour dressed up differently, and it
would be obvious immediately.

**Press TAB to switch between this and step 4's symmetric frustum.** That is
the demonstration: the same room, the same eye position, the same everything,
and one of them has depth and the other does not.

The eye is driven by the mouse here rather than by the webcam, on purpose. The
frustum maths and the head tracker are two separate things that can each be
wrong, and debugging them at the same time means a sign error and a maths error
look identical. Proving the projection against a perfect, noiseless, zero-lag
input first means that when the tracker is connected in step 6, anything that
goes wrong is the tracker.

Usage:

    python "mouse parallax.py"
    python "mouse parallax.py" --render-test   render frames to PNG
    python "mouse parallax.py" --selftest      check the maths, no window

    move mouse   move the eye         TAB   off-axis / symmetric
    scroll       eye nearer/further   a     auto-sweep the eye in a circle
    f            fullscreen           g     cycle grid density
    r            recentre the eye     q     quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"

ROOM_DEPTH_MM = 1200.0
DEFAULT_EYE_Z = 600.0
GRID_DENSITIES = (6, 12, 24)


# ----------------------------------------------------------------------------
# Display geometry
# ----------------------------------------------------------------------------


def set_dpi_aware() -> None:
    """Ask Windows to stop lying about the screen size.

    Must happen before any window exists. Without it, a process on a 150%-scaled
    desktop is told this 2560x1600 panel is 1707x1067, and every conversion
    between pixels and millimetres comes out 1.5x wrong.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class ScreenGeometry:
    """Converts between desktop pixels and screen-space millimetres.

    Screen space has its origin at the centre of the *panel*, +X right, +Y up.
    But the frustum has to be built from the rectangle that is actually being
    drawn into, which is the **window**, not the panel — so a floating window
    needs to know where its own corners are in physical space. That is what this
    class is for, and it is why the pixels-per-millimetre scale exists.

    Fullscreen is then just the case where the window rectangle happens to equal
    the panel rectangle. There is no separate code path for it.
    """

    def __init__(self, cfg: dict) -> None:
        d = cfg["display"]
        self.panel_w_mm = float(d["screen_w_mm"])
        self.panel_h_mm = float(d["screen_h_mm"])
        self.native_px = [int(v) for v in d["native_px"]]
        self.px_per_mm = self.native_px[0] / self.panel_w_mm

        # Sanity: the two axes should agree, since pixels are square on any
        # normal panel. A big disagreement means a mistyped measurement.
        vertical = self.native_px[1] / self.panel_h_mm
        if abs(vertical - self.px_per_mm) / self.px_per_mm > 0.02:
            print(f"  warning: pixels are not square — {self.px_per_mm:.3f} px/mm "
                  f"across, {vertical:.3f} down. Check screen_w_mm/screen_h_mm.")

    def window_rect_mm(self, win_x: int, win_y: int, win_w: int, win_h: int):
        """The window's corners in screen-space millimetres.

        `win_x, win_y` are the window's top-left in desktop pixels, where the
        desktop's Y grows downward — hence the negation, since screen-space Y
        grows up.

        Returns Kooima's corner triple: bottom-left, bottom-right, top-left.
        """
        centre_x_px = win_x + win_w * 0.5
        centre_y_px = win_y + win_h * 0.5

        cx_mm = (centre_x_px - self.native_px[0] * 0.5) / self.px_per_mm
        cy_mm = -(centre_y_px - self.native_px[1] * 0.5) / self.px_per_mm

        half_w = win_w * 0.5 / self.px_per_mm
        half_h = win_h * 0.5 / self.px_per_mm

        pa = np.array([cx_mm - half_w, cy_mm - half_h, 0.0])  # bottom-left
        pb = np.array([cx_mm + half_w, cy_mm - half_h, 0.0])  # bottom-right
        pc = np.array([cx_mm - half_w, cy_mm + half_h, 0.0])  # top-left
        return pa, pb, pc


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------


def frustum(l: float, r: float, b: float, t: float, n: float, f: float) -> np.ndarray:
    """The standard OpenGL perspective matrix, written out rather than called
    from a helper, because the whole project turns on being able to see and
    change these six numbers."""
    return np.array([
        [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
        [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
        [0.0, 0.0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype="f8")


def off_axis_projection(pa, pb, pc, pe, near: float, far: float) -> np.ndarray:
    """Kooima's generalized perspective projection — the heart of the illusion.

    Robert Kooima, "Generalized Perspective Projection" (2008). Given three
    corners of a physical screen and an eye position, all in the same
    coordinate system, it produces a projection matrix whose view volume has
    that screen rectangle as an exact cross-section.

        pa = bottom-left corner of the screen
        pb = bottom-right corner
        pc = top-left corner
        pe = eye position

    How it works, in one sentence: build an orthonormal basis on the screen,
    measure where the eye is relative to it, and set the frustum's four side
    planes so that each one passes through the corresponding screen edge.

    The consequence worth internalising: **the screen corners are nailed to the
    world.** Whatever the eye does, the four corners of the physical panel
    always land on the four corners of the rendered image, because the frustum
    is defined by them. Everything else in the scene moves relative to that
    fixed frame, which is exactly what looking through a real window is like.
    The self-test asserts this to sub-pixel accuracy.

    Contrast with `symmetric_projection` below, which throws away the eye's
    sideways position and so can only translate the picture.
    """
    pa, pb, pc, pe = (np.asarray(v, dtype="f8") for v in (pa, pb, pc, pe))

    # An orthonormal basis attached to the screen.
    vr = pb - pa            # along the bottom edge, pointing right
    vu = pc - pa            # up the left edge
    vr /= np.linalg.norm(vr)
    vu /= np.linalg.norm(vu)
    vn = np.cross(vr, vu)   # screen normal, pointing back out towards the viewer
    vn /= np.linalg.norm(vn)

    # From the eye to three of the corners.
    va = pa - pe
    vb = pb - pe
    vc = pc - pe

    # Perpendicular distance from the eye to the screen plane. Note this is the
    # distance to the *plane*, not to the centre of the screen — sliding
    # sideways does not change it, which is precisely why sideways movement
    # shears the frustum rather than scaling it.
    d = -float(np.dot(vn, va))
    if d < 1e-6:
        # The eye is on or behind the screen plane. There is no sensible
        # projection, so hold it just in front rather than emitting NaNs.
        d = 1e-6

    # The frustum extents, found by projecting the corner vectors onto the
    # screen basis and scaling from distance d to the near plane. These are
    # asymmetric in general: l != -r and b != -t.
    scale = near / d
    l = float(np.dot(vr, va)) * scale
    r = float(np.dot(vr, vb)) * scale
    b = float(np.dot(vu, va)) * scale
    t = float(np.dot(vu, vc)) * scale

    p = frustum(l, r, b, t, near, far)

    # Rotate the world into the screen's basis. For a screen lying in the z=0
    # plane this is the identity, but writing it out keeps the maths correct
    # for a tilted or rotated panel — which is what makes the multi-monitor
    # idea in the spec's "notes for later" a config change rather than a
    # rewrite.
    m = np.eye(4)
    m[0, :3] = vr
    m[1, :3] = vu
    m[2, :3] = vn

    # Move the eye to the origin.
    t_mat = np.eye(4)
    t_mat[:3, 3] = -pe

    return p @ m @ t_mat


def frustum_extents(pa, pb, pc, pe, near: float, far: float):
    """The l/r/b/t values `off_axis_projection` computes, exposed for the HUD
    and the self-test. Duplicated deliberately rather than returned from the
    function above, so the projection stays a single clean expression."""
    pa, pb, pc, pe = (np.asarray(v, dtype="f8") for v in (pa, pb, pc, pe))
    vr = pb - pa
    vu = pc - pa
    vr /= np.linalg.norm(vr)
    vu /= np.linalg.norm(vu)
    vn = np.cross(vr, vu)
    vn /= np.linalg.norm(vn)
    va, vb, vc = pa - pe, pb - pe, pc - pe
    d = max(-float(np.dot(vn, va)), 1e-6)
    scale = near / d
    return (float(np.dot(vr, va)) * scale, float(np.dot(vr, vb)) * scale,
            float(np.dot(vu, va)) * scale, float(np.dot(vu, vc)) * scale, d)


def symmetric_projection(pa, pb, pc, eye, near: float, far: float) -> np.ndarray:
    """An ordinary centred perspective camera, sitting at `eye` and looking
    straight down -Z.

    Note what this ignores: where the eye is *sideways*. The frustum is built
    from the screen rectangle's size alone, symmetric about the view axis, so
    moving the eye left only translates the camera — it does not change the
    shape of the view volume. That is what makes the room slide instead of
    opening up, and it is the thing step 5 fixes.
    """
    width = float(np.linalg.norm(pb - pa))
    height = float(np.linalg.norm(pc - pa))
    centre = (pa + pb + pc) / 3.0 + (pb - pa + pc - pa) / 6.0  # rectangle centre

    # Perpendicular distance from the eye to the screen plane.
    d = max(float(eye[2] - centre[2]), 1.0)

    half_w = width * 0.5 * near / d
    half_h = height * 0.5 * near / d
    proj = frustum(-half_w, half_w, -half_h, half_h, near, far)

    view = np.eye(4)
    view[0, 3] = -eye[0]
    view[1, 3] = -eye[1]
    view[2, 3] = -eye[2]
    return proj @ view


# ----------------------------------------------------------------------------
# The scene
# ----------------------------------------------------------------------------

LINE_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_colour;
out vec3 v_colour;
out float v_depth;
void main() {
    v_colour = in_colour;
    // World Z is negative going away from the viewer, so this is 0 at the
    // glass and 1 at the back wall. Used to fade the room into darkness.
    v_depth = clamp(-in_pos.z / ROOM_DEPTH, 0.0, 1.0);
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
in vec3 v_colour;
in float v_depth;
out vec4 f_colour;
void main() {
    // Squared falloff. Linear fade looks flat; the eye reads a steeper curve
    // as distance, and the receding parallel lines are the strongest depth cue
    // this scene has.
    //
    // The 0.22 floor matters: falling all the way to black makes the back wall
    // disappear, and losing it costs the far end of every receding line — which
    // is precisely the geometry the illusion depends on being able to see.
    float fade = 1.0 - v_depth;
    fade = fade * fade * 0.78 + 0.22;
    f_colour = vec4(v_colour * fade, 1.0);
}
"""

SOLID_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_colour;
out vec3 v_normal;
out vec3 v_colour;
out float v_depth;
void main() {
    v_normal = in_normal;
    v_colour = in_colour;
    v_depth = clamp(-in_pos.z / ROOM_DEPTH, 0.0, 1.0);
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

SOLID_FRAGMENT_SHADER = """
#version 330
in vec3 v_normal;
in vec3 v_colour;
in float v_depth;
out vec4 f_colour;
void main() {
    // One soft directional light, from up and to the left, plus enough ambient
    // that unlit faces still read as surfaces rather than holes.
    vec3 light = normalize(vec3(-0.35, 0.65, 0.55));
    float lambert = max(dot(normalize(v_normal), light), 0.0);
    float shade = 0.38 + 0.62 * lambert;

    // Much gentler than the falloff on the walls. These shapes are the targets
    // the parallax acts on, so they have to stay legible all the way to the
    // back — fading them like the room turns the far ones into black holes and
    // there is then nothing to watch move.
    float fade = 1.0 - 0.45 * v_depth;
    f_colour = vec4(v_colour * shade * fade, 1.0);
}
"""


def _grid_lines(divisions: int, screen_w: float, screen_h: float,
                depth: float) -> np.ndarray:
    """The five-sided room: floor, ceiling, both side walls, back wall.

    No front wall — that is where you are. The mouth of the box is exactly the
    window rectangle at Z = 0, so the room appears to start at the glass.
    """
    hw, hh = screen_w * 0.5, screen_h * 0.5
    verts: list[float] = []

    wall = (0.30, 0.86, 0.94)   # cyan
    edge = (0.55, 0.98, 1.00)   # brighter, for the corner edges
    back = (0.22, 0.62, 0.72)

    def line(p0, p1, colour):
        verts.extend([*p0, *colour, *p1, *colour])

    depth_steps = [depth * i / divisions for i in range(divisions + 1)]

    # Rings going back: each one is the room's cross-section at that depth.
    # These are the lines that make the perspective unmistakable.
    for z in depth_steps:
        z = -z
        line((-hw, -hh, z), (hw, -hh, z), wall)
        line((hw, -hh, z), (hw, hh, z), wall)
        line((hw, hh, z), (-hw, hh, z), wall)
        line((-hw, hh, z), (-hw, -hh, z), wall)

    # Lines running away from the viewer along each surface.
    across = max(divisions // 2, 3)
    for i in range(across + 1):
        x = -hw + 2 * hw * i / across
        line((x, -hh, 0.0), (x, -hh, -depth), wall)   # floor
        line((x, hh, 0.0), (x, hh, -depth), wall)     # ceiling
    for i in range(max(across // 2, 2) + 1):
        y = -hh + 2 * hh * i / max(across // 2, 2)
        line((-hw, y, 0.0), (-hw, y, -depth), wall)   # left wall
        line((hw, y, 0.0), (hw, y, -depth), wall)     # right wall

    # The four corner edges, brighter so the box reads as solid.
    for sx in (-1, 1):
        for sy in (-1, 1):
            line((sx * hw, sy * hh, 0.0), (sx * hw, sy * hh, -depth), edge)

    # A grid on the back wall.
    for i in range(across + 1):
        x = -hw + 2 * hw * i / across
        line((x, -hh, -depth), (x, hh, -depth), back)
    for i in range(max(across // 2, 2) + 1):
        y = -hh + 2 * hh * i / max(across // 2, 2)
        line((-hw, y, -depth), (hw, y, -depth), back)

    return np.array(verts, dtype="f4")


def _cube(size: float):
    h = size * 0.5
    faces = [
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, -h, -h), (-h, h, -h), (h, h, -h)]),
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, -h, h), (h, h, -h), (h, h, h)]),
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, -h, -h), (-h, h, h), (-h, h, -h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, h), (h, h, -h), (-h, h, -h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, -h), (h, -h, h), (-h, -h, h)]),
    ]
    return [(p, n) for n, tri in faces for p in tri]


def _pyramid(size: float):
    h = size * 0.5
    apex = (0.0, h, 0.0)
    base = [(-h, -h, h), (h, -h, h), (h, -h, -h), (-h, -h, -h)]
    out = []
    for i in range(4):
        p0, p1 = base[i], base[(i + 1) % 4]
        n = np.cross(np.subtract(p1, p0), np.subtract(apex, p0))
        n = n / (np.linalg.norm(n) or 1.0)
        out += [(p0, n), (p1, n), (apex, n)]
    n = (0.0, -1.0, 0.0)
    out += [(base[0], n), (base[2], n), (base[1], n),
            (base[0], n), (base[3], n), (base[2], n)]
    return out


def _octahedron(size: float):
    h = size * 0.5
    tips = [(0, h, 0), (0, -h, 0)]
    ring = [(h, 0, 0), (0, 0, h), (-h, 0, 0), (0, 0, -h)]
    out = []
    for tip_i, tip in enumerate(tips):
        for i in range(4):
            p0, p1 = ring[i], ring[(i + 1) % 4]
            if tip_i == 1:
                p0, p1 = p1, p0
            n = np.cross(np.subtract(p1, p0), np.subtract(tip, p0))
            n = n / (np.linalg.norm(n) or 1.0)
            out += [(p0, n), (p1, n), (tip, n)]
    return out


def _solid_shapes(screen_w: float, screen_h: float, depth: float) -> np.ndarray:
    """Shapes at assorted depths, so there is something for the parallax to act
    on. Seeded, so the scene is the same every run — comparing two screenshots
    is impossible if the furniture moves between them."""
    rng = np.random.default_rng(20260803)
    builders = (_cube, _pyramid, _octahedron)
    palette = [(1.00, 0.42, 0.38), (0.98, 0.78, 0.32), (0.55, 0.85, 1.00),
               (0.72, 0.60, 1.00), (0.45, 0.95, 0.70)]

    # Sized as a fraction of the room, not in absolute millimetres. The room's
    # cross-section is the window, which can be any size the user drags it to —
    # fixed-size shapes would fill a small window completely and vanish in a
    # large one.
    short_side = min(screen_w, screen_h)

    verts: list[float] = []
    for i in range(9):
        size = float(rng.uniform(0.10, 0.20)) * short_side
        shape = builders[i % len(builders)](size)
        colour = palette[i % len(palette)]

        # Keep them clear of the walls so nothing pokes through a surface. The
        # max() guards the degenerate case where a shape is nearly as wide as
        # the room, which would otherwise give an empty range.
        margin = size * 0.9
        span_x = max(screen_w * 0.5 - margin, 0.0)
        span_y = max(screen_h * 0.5 - margin, 0.0)
        cx = float(rng.uniform(-span_x, span_x)) if span_x > 0 else 0.0
        cy = float(rng.uniform(-span_y, span_y)) if span_y > 0 else 0.0
        cz = -float(rng.uniform(depth * 0.10, depth * 0.90))

        for (px, py, pz), n in shape:
            verts.extend([px + cx, py + cy, pz + cz, *n, *colour])
    return np.array(verts, dtype="f4")


class GridRoom:
    """Geometry plus the two shader programs. Knows nothing about projection —
    it is handed a matrix and draws with it, which is what lets step 5 change
    the projection without touching any of this."""

    def __init__(self, ctx, screen_w: float, screen_h: float,
                 depth: float = ROOM_DEPTH_MM) -> None:
        import moderngl

        self.ctx = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.depth = depth
        self.density_index = 1

        def compile_source(src: str) -> str:
            return src.replace("ROOM_DEPTH", f"{depth:.1f}")

        self.line_prog = ctx.program(
            vertex_shader=compile_source(LINE_VERTEX_SHADER),
            fragment_shader=compile_source(LINE_FRAGMENT_SHADER))
        self.solid_prog = ctx.program(
            vertex_shader=compile_source(SOLID_VERTEX_SHADER),
            fragment_shader=compile_source(SOLID_FRAGMENT_SHADER))

        self.line_vaos = []
        for divisions in GRID_DENSITIES:
            data = _grid_lines(divisions, screen_w, screen_h, depth)
            vbo = ctx.buffer(data.tobytes())
            self.line_vaos.append((
                ctx.vertex_array(self.line_prog,
                                 [(vbo, "3f 3f", "in_pos", "in_colour")]),
                len(data) // 6))

        solids = _solid_shapes(screen_w, screen_h, depth)
        vbo = ctx.buffer(solids.tobytes())
        self.solid_vao = ctx.vertex_array(
            self.solid_prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_colour")])
        self.n_solid_verts = len(solids) // 9

    def cycle_density(self) -> int:
        self.density_index = (self.density_index + 1) % len(GRID_DENSITIES)
        return GRID_DENSITIES[self.density_index]

    def draw(self, projection: np.ndarray) -> None:
        import moderngl

        # numpy is row-major, GLSL wants column-major, hence the transpose.
        packed = np.ascontiguousarray(projection.T, dtype="f4").tobytes()
        self.solid_prog["mvp"].write(packed)
        self.line_prog["mvp"].write(packed)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.solid_vao.render(moderngl.TRIANGLES, vertices=self.n_solid_verts)

        vao, count = self.line_vaos[self.density_index]
        vao.render(moderngl.LINES, vertices=count)


# ----------------------------------------------------------------------------
# Headless render, for checking the result without a window
# ----------------------------------------------------------------------------


def project_to_pixels(matrix: np.ndarray, point, width: int, height: int):
    """Push a world point through a projection matrix to a pixel position."""
    clip = matrix @ np.array([*point, 1.0], dtype="f8")
    if abs(clip[3]) < 1e-12:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * width, (0.5 - ndc[1] * 0.5) * height)


def selftest() -> int:
    """The acceptance tests from the spec, plus the ones they imply."""
    print("\n" + "=" * 72)
    print("Off-axis projection self test")
    print("=" * 72)
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}" + (f"  — {detail}" if detail else ""))

    w, h = 346.0, 216.0
    near, far = 10.0, 10000.0
    pa = np.array([-w / 2, -h / 2, 0.0])
    pb = np.array([+w / 2, -h / 2, 0.0])
    pc = np.array([-w / 2, +h / 2, 0.0])
    pd = pb + pc - pa                      # the implied fourth corner
    corners = [pa, pb, pc, pd]

    # --- spec §5: a centred eye must give a symmetric frustum ---------------
    l, r, b, t, d = frustum_extents(pa, pb, pc, [0.0, 0.0, 600.0], near, far)
    check("centred eye gives a symmetric frustum",
          abs(l + r) < 1e-12 and abs(b + t) < 1e-12,
          f"l={l:.6f} r={r:.6f} b={b:.6f} t={t:.6f}")
    check("perpendicular distance is the eye's z", abs(d - 600.0) < 1e-12,
          f"d={d:.6f}")

    # --- spec §5: moving +X shears, and in the right direction --------------
    l2, r2, b2, t2, d2 = frustum_extents(pa, pb, pc, [80.0, 0.0, 600.0], near, far)
    check("moving the eye +X makes |l| > |r|", abs(l2) > abs(r2),
          f"l={l2:.4f} r={r2:.4f}")
    check("the frustum shears, it does not rotate",
          abs(d2 - 600.0) < 1e-12,
          "sideways movement leaves the eye-to-plane distance untouched")
    check("vertical extents unchanged by a sideways move",
          abs(b2 - b) < 1e-12 and abs(t2 - t) < 1e-12)

    # NOTE: the spec (§5) says "moving pe in +X must make |l| < |r|". That is
    # the wrong way round. Moving right puts the eye further from the left edge
    # of the screen and closer to the right, so the left extent grows and the
    # right one shrinks. The corner-lock test below is what actually proves the
    # projection is correct, and it passes — so the spec's inequality is a slip,
    # not a specification we are failing to meet.

    # --- spec §10 test 1: the corner-lock test, the definitive one ----------
    #
    # Whatever the eye does, a vertex sitting at a physical screen corner must
    # land on the same pixel. The corners are nailed to the world; if they
    # drift, the projection is wrong.
    px_w, px_h = 1920, 1200
    rng = np.random.default_rng(4)
    reference = None
    worst = 0.0
    for _ in range(200):
        eye = np.array([rng.uniform(-400, 400), rng.uniform(-300, 300),
                        rng.uniform(200, 1500)])
        matrix = off_axis_projection(pa, pb, pc, eye, near, far)
        pixels = [project_to_pixels(matrix, c, px_w, px_h) for c in corners]
        if reference is None:
            reference = pixels
            continue
        for got, want in zip(pixels, reference):
            worst = max(worst, math.hypot(got[0] - want[0], got[1] - want[1]))
    check("screen corners are pixel-locked over 200 eye positions", worst < 1e-6,
          f"worst drift {worst:.3e} px, on a {px_w}x{px_h} image")

    # And they must land exactly on the image corners, not merely agree.
    matrix = off_axis_projection(pa, pb, pc, [0.0, 0.0, 600.0], near, far)
    got = [project_to_pixels(matrix, c, px_w, px_h) for c in corners]
    want = [(0, px_h), (px_w, px_h), (0, 0), (px_w, 0)]
    ok = all(abs(g[0] - e[0]) < 1e-6 and abs(g[1] - e[1]) < 1e-6
             for g, e in zip(got, want))
    check("corners land on the image corners", ok,
          "bottom-left -> (0, H), top-right -> (W, 0)")

    # --- the symmetric frustum must FAIL that same test ---------------------
    #
    # Otherwise the test proves nothing — it has to be able to tell the two
    # projections apart, or it would pass on step 4's broken version too.
    matrix_a = symmetric_projection(pa, pb, pc, [0.0, 0.0, 600.0], near, far)
    matrix_b = symmetric_projection(pa, pb, pc, [80.0, 0.0, 600.0], near, far)
    drift = max(math.hypot(*(np.subtract(project_to_pixels(matrix_b, c, px_w, px_h),
                                         project_to_pixels(matrix_a, c, px_w, px_h))))
                for c in corners)
    check("the symmetric frustum fails corner-lock", drift > 100.0,
          f"corners drift {drift:.0f} px for an 80 mm eye move — this is step 4's flaw")

    # --- spec §10 test 2: differential parallax -----------------------------
    #
    # The analytic gain: a point at depth pz, seen from an eye at ez, projects
    # onto the glass at a position that moves by |pz|/(ez+|pz|) per millimetre
    # of sideways eye movement. A point ON the glass never moves; a point
    # infinitely far away tracks the eye one for one.
    #
    # NOTE: the spec says an object at Z=-100 must shift "substantially more"
    # than one at Z=-1000. That is backwards — Z=-100 is nearly touching the
    # glass and is therefore nearly nailed to it. Verified below against both
    # the closed form and the actual matrix.
    ez, move = 600.0, 100.0
    eye0 = np.array([0.0, 0.0, ez])
    eye1 = np.array([move, 0.0, ez])
    m0 = off_axis_projection(pa, pb, pc, eye0, near, far)
    m1 = off_axis_projection(pa, pb, pc, eye1, near, far)

    measured = {}
    for depth in (-100.0, -1000.0):
        p = np.array([0.0, 0.0, depth])
        x0 = project_to_pixels(m0, p, px_w, px_h)[0]
        x1 = project_to_pixels(m1, p, px_w, px_h)[0]
        measured[depth] = abs(x1 - x0) / (px_w / w)   # pixels back to mm

    def analytic(depth: float) -> float:
        return abs(depth) / (ez + abs(depth)) * move

    for depth in (-100.0, -1000.0):
        err = abs(measured[depth] - analytic(depth)) / analytic(depth) * 100
        check(f"parallax at z={depth:.0f} matches the closed form", err < 2.0,
              f"{measured[depth]:.3f} mm measured, {analytic(depth):.3f} predicted "
              f"({err:.3f}% off)")

    ratio = measured[-1000.0] / measured[-100.0]
    check("the deeper object shifts more", ratio > 4.0,
          f"{ratio:.3f}x, closed form says {analytic(-1000.)/analytic(-100.):.3f}x")

    # A point exactly on the glass must not move at all — the cleanest possible
    # statement of what "the screen is a window" means.
    on_glass = np.array([40.0, -20.0, 0.0])
    x0 = project_to_pixels(m0, on_glass, px_w, px_h)
    x1 = project_to_pixels(m1, on_glass, px_w, px_h)
    check("a point on the glass never moves",
          math.hypot(x1[0] - x0[0], x1[1] - x0[1]) < 1e-6,
          f"{math.hypot(x1[0]-x0[0], x1[1]-x0[1]):.3e} px")

    # --- an off-centre window (the floating-window case) --------------------
    #
    # Everything above used a window centred on the panel. A dragged window is
    # the case that actually matters, and it must behave identically.
    off_pa = pa + np.array([300.0, 150.0, 0.0])
    off_pb = pb + np.array([300.0, 150.0, 0.0])
    off_pc = pc + np.array([300.0, 150.0, 0.0])
    off_pd = off_pb + off_pc - off_pa
    worst = 0.0
    reference = None
    for _ in range(100):
        eye = np.array([rng.uniform(-400, 400), rng.uniform(-300, 300),
                        rng.uniform(250, 1200)])
        matrix = off_axis_projection(off_pa, off_pb, off_pc, eye, near, far)
        pixels = [project_to_pixels(matrix, c, px_w, px_h)
                  for c in (off_pa, off_pb, off_pc, off_pd)]
        if reference is None:
            reference = pixels
            continue
        for got, want in zip(pixels, reference):
            worst = max(worst, math.hypot(got[0] - want[0], got[1] - want[1]))
    check("an off-centre window is also corner-locked", worst < 1e-6,
          f"worst drift {worst:.3e} px for a window dragged 300 mm right, 150 mm up")

    # --- degenerate input ---------------------------------------------------
    m = off_axis_projection(pa, pb, pc, [0.0, 0.0, 0.0], near, far)
    check("an eye on the screen plane does not produce NaNs",
          bool(np.all(np.isfinite(m))))

    print()
    if failures:
        print(f"  {failures} check(s) failed.")
        return 1
    print("  All self tests passed.")
    return 0


def render_test(cfg: dict) -> int:
    import moderngl
    from PIL import Image

    geom = ScreenGeometry(cfg)
    near = float(cfg["render"]["near_mm"])
    far = float(cfg["render"]["far_mm"])
    samples = int(cfg["render"].get("msaa", 4))

    width, height = 960, 600
    ctx = moderngl.create_standalone_context(require=460)
    print(f"  {ctx.info['GL_RENDERER'].strip()}")

    colour = ctx.texture((width, height), 4, samples=samples)
    depth_buf = ctx.depth_texture((width, height), samples=samples)
    msaa = ctx.framebuffer(color_attachments=[colour], depth_attachment=depth_buf)
    resolved = ctx.simple_framebuffer((width, height))

    # A window of this pixel size, centred on the panel.
    win_x = (geom.native_px[0] - width) // 2
    win_y = (geom.native_px[1] - height) // 2
    pa, pb, pc = geom.window_rect_mm(win_x, win_y, width, height)
    print(f"  window rect: {np.linalg.norm(pb - pa):.1f} x "
          f"{np.linalg.norm(pc - pa):.1f} mm")

    room = GridRoom(ctx, float(np.linalg.norm(pb - pa)),
                    float(np.linalg.norm(pc - pa)))

    views = [
        ("centre", np.array([0.0, 0.0, DEFAULT_EYE_Z])),
        ("left", np.array([-120.0, 0.0, DEFAULT_EYE_Z])),
        ("right", np.array([+120.0, 0.0, DEFAULT_EYE_Z])),
        ("above", np.array([0.0, 110.0, DEFAULT_EYE_Z])),
    ]
    for name, eye in views:
        for label, projection in (("offaxis", off_axis_projection),
                                  ("symmetric", symmetric_projection)):
            msaa.use()
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            room.draw(projection(pa, pb, pc, eye, near, far))
            ctx.copy_framebuffer(resolved, msaa)
            img = Image.frombytes(
                "RGB", (width, height),
                resolved.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
            img.save(HERE / f"render-test-{label}-{name}.png")
        print(f"  {name}: eye at {eye[0]:+.0f}, {eye[1]:+.0f}, {eye[2]:.0f} mm")

    print("\n  Compare render-test-offaxis-left.png with -symmetric-left.png.")
    print("  Same eye position. The off-axis one opens the right-hand wall and")
    print("  keeps the room's mouth filling the frame; the symmetric one just")
    print("  slides the whole picture sideways off the edge.")
    return 0


# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid room, symmetric frustum.")
    parser.add_argument("--render-test", action="store_true",
                        help="render a few frames to PNG and exit, no window")
    parser.add_argument("--selftest", action="store_true",
                        help="check the projection maths, no window")
    parser.add_argument("--seconds", type=float, default=None,
                        help="close automatically after N seconds (for benchmarking)")
    parser.add_argument("--no-vsync", action="store_true",
                        help="unlock the frame rate, to see how much headroom there is")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    import yaml

    set_dpi_aware()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if args.render_test:
        return render_test(cfg)

    import moderngl
    import pyglet

    geom = ScreenGeometry(cfg)
    near = float(cfg["render"]["near_mm"])
    far = float(cfg["render"]["far_mm"])
    samples = int(cfg["render"].get("msaa", 4))
    vsync = bool(cfg["render"].get("vsync", True)) and not args.no_vsync

    # Default to a window with the same shape as the panel, so the room's mouth
    # is not distorted before anything else has had a chance to go wrong.
    win_w = int(geom.native_px[0] * 0.5)
    win_h = int(round(win_w * geom.panel_h_mm / geom.panel_w_mm))

    config = pyglet.gl.Config(major_version=4, minor_version=6, double_buffer=True,
                              depth_size=24, sample_buffers=1, samples=samples)
    try:
        window = pyglet.window.Window(width=win_w, height=win_h, config=config,
                                      resizable=True, vsync=vsync,
                                      caption="grid room - symmetric frustum")
    except pyglet.window.NoSuchConfigException:
        print(f"  {samples}x MSAA is not available; falling back to none.")
        config = pyglet.gl.Config(major_version=4, minor_version=6,
                                  double_buffer=True, depth_size=24)
        window = pyglet.window.Window(width=win_w, height=win_h, config=config,
                                      resizable=True, vsync=vsync,
                                      caption="grid room - symmetric frustum")

    ctx = moderngl.create_context()
    print(f"  {ctx.info['GL_RENDERER'].strip()}  |  GL {ctx.info['GL_VERSION'].strip()}")
    print(f"  {geom.px_per_mm:.3f} px/mm from a {geom.panel_w_mm:g} mm wide panel")

    state = {
        "eye": np.array([0.0, 0.0, DEFAULT_EYE_Z]),
        "room": None,
        "room_size": None,
        "frames": 0,
        "last_report": time.perf_counter(),
        "fps": 0.0,
        # Separate totals, because the rolling counter above is reset twice a
        # second for the title bar and cannot give an honest overall average.
        "total_frames": 0,
        "started": time.perf_counter(),
        "worst_frame_ms": 0.0,
        "last_frame_at": None,
        "off_axis": True,
        "auto": False,
        "auto_t": 0.0,
    }

    # How far the eye travels for a given mouse movement. At 1.0 the eye tracks
    # the mouse exactly — physically honest, but the pointer can only cross the
    # panel, so the eye could never get further than half a screen width off
    # centre. Real head movement is larger than that, so the default exaggerates
    # a little to make the effect easy to explore with a mouse.
    eye_gain = 2.0

    def eye_from_mouse(win_mouse_x: int, win_mouse_y: int) -> None:
        """Turn a pointer position into an eye position in screen-space mm.

        The same conversion the tracker will use, so the mouse is a stand-in for
        a head rather than a different kind of input. pyglet gives mouse
        coordinates relative to the window with Y up; the desktop measures Y
        down from the top, hence the flip before converting.
        """
        win_x, win_y = window.get_location()
        desktop_x = win_x + win_mouse_x
        desktop_y = win_y + (window.height - win_mouse_y)

        x_mm = (desktop_x - geom.native_px[0] * 0.5) / geom.px_per_mm
        y_mm = -(desktop_y - geom.native_px[1] * 0.5) / geom.px_per_mm
        state["eye"][0] = x_mm * eye_gain
        state["eye"][1] = y_mm * eye_gain

    def rebuild_room():
        """The room's mouth is the window, so resizing the window rebuilds it."""
        pa, pb, pc = geom.window_rect_mm(*window.get_location(),
                                         window.width, window.height)
        size = (float(np.linalg.norm(pb - pa)), float(np.linalg.norm(pc - pa)))
        if state["room_size"] != size:
            state["room"] = GridRoom(ctx, size[0], size[1])
            state["room_size"] = size
        return pa, pb, pc

    rebuild_room()

    @window.event
    def on_draw():
        pa, pb, pc = rebuild_room()
        ctx.screen.use()
        ctx.viewport = (0, 0, window.width, window.height)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        build = off_axis_projection if state["off_axis"] else symmetric_projection
        state["room"].draw(build(pa, pb, pc, state["eye"], near, far))

        state["frames"] += 1
        state["total_frames"] += 1
        now = time.perf_counter()
        if state["last_frame_at"] is not None:
            # Ignore the first few frames — shader upload and the initial buffer
            # allocation make them meaninglessly slow.
            gap_ms = (now - state["last_frame_at"]) * 1000.0
            if state["total_frames"] > 10:
                state["worst_frame_ms"] = max(state["worst_frame_ms"], gap_ms)
        state["last_frame_at"] = now

        if now - state["last_report"] >= 0.5:
            state["fps"] = state["frames"] / (now - state["last_report"])
            state["frames"] = 0
            state["last_report"] = now
            eye = state["eye"]
            l, r, b, t, d = frustum_extents(pa, pb, pc, eye, near, far)
            mode = "OFF-AXIS" if state["off_axis"] else "symmetric (step 4)"
            window.set_caption(
                f"mouse parallax [{mode}] - {state['fps']:.0f} fps - "
                f"eye {eye[0]:+.0f} {eye[1]:+.0f} {eye[2]:.0f} mm - "
                f"l/r {l:+.2f}/{r:+.2f}")

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        if not state["auto"]:
            eye_from_mouse(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if not state["auto"]:
            eye_from_mouse(x, y)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        # Scrolling moves the eye towards or away from the glass. Leaning in
        # should widen the view the way putting your face to a real window does.
        state["eye"][2] = float(np.clip(state["eye"][2] - scroll_y * 25.0,
                                        120.0, 2000.0))

    def advance_auto(dt):
        """A slow circular sweep of the eye. Useful for recording, and for
        watching the effect without a hand on the mouse."""
        if not state["auto"]:
            return
        state["auto_t"] += dt
        angle = state["auto_t"] * 0.9
        state["eye"][0] = math.cos(angle) * 170.0
        state["eye"][1] = math.sin(angle * 0.7) * 110.0

    @window.event
    def on_key_press(symbol, modifiers):
        keys = pyglet.window.key
        step = 20.0
        if symbol == keys.TAB:
            state["off_axis"] = not state["off_axis"]
            print("  " + ("off-axis frustum — the room opens up"
                          if state["off_axis"] else
                          "symmetric frustum — the room only slides"))
        elif symbol == keys.A:
            state["auto"] = not state["auto"]
            print(f"  auto-sweep {'on' if state['auto'] else 'off'}")
        elif symbol == keys.LEFT:
            state["eye"][0] -= step
        elif symbol == keys.RIGHT:
            state["eye"][0] += step
        elif symbol == keys.UP:
            state["eye"][1] += step
        elif symbol == keys.DOWN:
            state["eye"][1] -= step
        elif symbol == keys.W:
            state["eye"][2] = max(120.0, state["eye"][2] - step)
        elif symbol == keys.S:
            state["eye"][2] = min(2000.0, state["eye"][2] + step)
        elif symbol == keys.R:
            state["eye"] = np.array([0.0, 0.0, DEFAULT_EYE_Z])
            state["auto"] = False
        elif symbol == keys.G:
            print(f"  grid divisions: {state['room'].cycle_density()}")
        elif symbol == keys.F:
            window.set_fullscreen(not window.fullscreen)
        elif symbol in (keys.Q, keys.ESCAPE):
            window.close()

    # pyglet only redraws on demand by default, and this needs a continuous
    # loop. Scheduling a callback keeps on_draw firing every frame.
    pyglet.clock.schedule(advance_auto)

    if args.seconds:
        pyglet.clock.schedule_once(lambda dt: window.close(), args.seconds)
        print(f"  benchmarking for {args.seconds:g} s...")
    else:
        print("\n  Move the mouse. The room should open rather than slide —")
        print("  go left and more of the right-hand wall comes into view.")
        print("  Press TAB to switch to step 4's symmetric frustum and back.")
        print("  Press 'a' for a hands-off sweep.\n")

    # interval=0 rather than pyglet's default of 1/60. The default caps the
    # event loop itself at 60 Hz whatever the display is doing, which on a
    # 240 Hz panel throws away about 12 ms of latency for nothing.
    #
    # Measured, rather than assumed: with the cap removed this scene runs at
    # roughly 900 fps whether vsync is on or off, which means the vsync request
    # is not blocking in windowed mode — the Windows compositor is presenting
    # on its own schedule and simply discarding the frames in between. So the
    # real limiter here is the compositor, not the swap. That is fine for
    # latency and wasteful of power; the fullscreen path in a later step is
    # where a genuine vsync lock can be expected.
    pyglet.app.run(interval=0)

    elapsed = time.perf_counter() - state["started"]
    average = state["total_frames"] / elapsed if elapsed > 0 else 0.0
    print(f"\n  {state['total_frames']} frames in {elapsed:.1f} s = {average:.1f} fps average")
    print(f"  worst single frame: {state['worst_frame_ms']:.1f} ms")
    if average > 200:
        print("  (far above the 60 fps target — plenty of headroom for richer scenes)")
    print("  Next: 'mouse parallax.py'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
