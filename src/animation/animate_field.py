"""Manim building blocks for 3D visualization of the solver's objects.

Optional: this module imports manim, which is an extra
(`pip install -e ".[animation]"`). Nothing in `src/` outside this package
imports it.

The scenes that actually get rendered into `assets/` live in
`scripts/render_animations.py`; this module holds the reusable pieces.

Everything works in the unit cube [0,1]^3 and maps into manim's camera frame
through `to_camera`, so all the mobjects here share one coordinate convention.
"""

import numpy as np
from manim import *  # noqa: F401,F403

# The unit cube maps to a cube of this side length centred on the origin.
CAMERA_SCALE = 7.0

CURVE_COLOR = ManimColor("#e8482c")
BACKGROUND = ManimColor("#ffffff")


def to_camera(points, scale: float = CAMERA_SCALE):
    """Unit-cube coordinates -> manim camera coordinates."""
    return (np.asarray(points, dtype=float) - 0.5) * scale


def from_camera(points, scale: float = CAMERA_SCALE):
    """Inverse of `to_camera`."""
    return np.asarray(points, dtype=float) / scale + 0.5


def colormap_color(value, cmap="viridis"):
    """Sample a matplotlib colormap and return a manim colour."""
    return _colormap_table(cmap)[int(np.clip(value, 0.0, 1.0) * 255)]


_TABLES = {}


def _colormap_table(cmap, levels=256):
    """Pre-quantized colour table.

    A mesh has thousands of triangles and an animation has thousands of frames,
    so looking the colormap up per triangle is worth avoiding.
    """
    if cmap not in _TABLES:
        import matplotlib

        lut = matplotlib.colormaps[cmap](np.linspace(0.0, 1.0, levels))
        _TABLES[cmap] = [ManimColor(rgb_to_hex(tuple(row[:3]))) for row in lut]
    return _TABLES[cmap]


def curve_mobject(polylines, color=CURVE_COLOR, width=3.5, scale=CAMERA_SCALE):
    """Draw one or more closed boundary loops."""
    from src import curves as _curves

    group = VGroup()
    for poly in _curves.as_polylines(polylines, 400):
        pts = to_camera(np.vstack([poly, poly[:1]]), scale)
        group.add(VMobject().set_points_as_corners(pts).set_stroke(color, width=width))
    return group


def mesh_mobject(
    vertices,
    faces,
    cmap="viridis",
    vmin=None,
    vmax=None,
    opacity=0.9,
    stroke_width=0.0,
    scale=CAMERA_SCALE,
):
    """A triangle mesh as a VGroup of shaded triangles.

    Coloured by the height of each triangle's centroid against a *fixed* range,
    so a flat surface renders as one flat colour rather than a bullseye made of
    sub-cell wobble. Pass vmin/vmax to keep the mapping stable across frames of
    an animation.
    """
    verts = to_camera(vertices, scale)
    if len(faces) == 0:
        return VGroup()

    centroid_z = verts[faces][:, :, 2].mean(axis=1)
    lo = centroid_z.min() if vmin is None else to_camera([[0, 0, vmin]], scale)[0, 2]
    hi = centroid_z.max() if vmax is None else to_camera([[0, 0, vmax]], scale)[0, 2]
    span = max(hi - lo, 1e-9)

    group = VGroup()
    for tri, z in zip(verts[faces], centroid_z):
        group.add(
            ThreeDVMobject()
            .set_points_as_corners([tri[0], tri[1], tri[2], tri[0]])
            .set_fill(colormap_color((z - lo) / span, cmap), opacity=opacity)
            .set_stroke(width=stroke_width)
        )
    return group


def field_sampler(grid, field, normalize=True, speed=1.0, scale=CAMERA_SCALE):
    """Build the `func` that manim's StreamLines integrates.

    With `normalize=True` the streamlines advance at constant speed, which is the
    usual way to draw field *lines*: the near-singular spike of a Dirac-delta
    form would otherwise make every streamline outside its support sit still.
    Magnitude is better carried by colour -- see `magnitude_scheme`.
    """
    field = np.asarray(field)

    def func(camera_point):
        p = np.clip(from_camera(camera_point, scale), 0.0, 1.0 - 1e-9)
        value = field[tuple(grid.position_to_index(p))]
        if normalize:
            norm = np.linalg.norm(value)
            if norm < 1e-12:
                return np.zeros(3)
            value = value / norm
        return np.asarray(value, dtype=float) * speed

    return func


def magnitude_scheme(grid, field, scale=CAMERA_SCALE):
    """A `color_scheme` callable giving |field| at a camera point."""
    magnitude = np.linalg.norm(np.asarray(field), axis=-1)

    def scheme(camera_point):
        p = np.clip(from_camera(camera_point, scale), 0.0, 1.0 - 1e-9)
        return float(magnitude[tuple(grid.position_to_index(p))])

    return scheme


def build_vector_field_scene(
    grid,
    vector_field,
    curve_function=None,
    surface_function=None,
    curve_color=CURVE_COLOR,
    surface_color=BLACK,
    streak_color=None,
    background_color=BACKGROUND,
    streak_time=2.0,
    streak_opacity=1.0,
    dt=0.05,
    streak_scaling: float = 1.0,
    normalize_field: bool = True,
    colors=None,
    rotation_rate: float = 0.12,
    run_time: float = 10.0,
    seed_spacing: float = 0.9,
    title: str = None,
):
    """A ThreeDScene showing a 1- or 2-form as animated streamlines.

    `vector_field` is a plain (N, N, N, 3) array -- `solution.eta`,
    `solution.eta_0`, or any field on the same grid.
    """
    field = np.asarray(vector_field)
    magnitude = np.linalg.norm(field, axis=-1)
    func = field_sampler(grid, field, normalize=normalize_field, speed=streak_scaling)
    scheme = magnitude_scheme(grid, field)
    half = CAMERA_SCALE / 2

    class DiracStreamLineScene(ThreeDScene):
        def construct(self):
            self.camera.background_color = background_color
            self.set_camera_orientation(phi=68 * DEGREES, theta=-55 * DEGREES, zoom=0.9)

            if curve_function is not None:
                self.add(curve_mobject(curve_function, color=curve_color))

            if surface_function is not None:
                self.add(
                    Surface(
                        lambda u, v: to_camera(surface_function(u, v)),
                        u_range=(0, 1),
                        v_range=(0, 1),
                        fill_opacity=0.8,
                        fill_color=surface_color,
                        checkerboard_colors=[surface_color],
                    )
                )

            # Seed spacing drives both render time and file size: the seed count
            # is cubic in it, and thousands of thin moving strokes are close to
            # worst case for a video encoder. It is also a legibility knob --
            # a dense seeding just reads as fog.
            stream_kwargs = dict(
                func=func,
                x_range=[-half, half, seed_spacing],
                y_range=[-half, half, seed_spacing],
                z_range=[-half, half, seed_spacing],
                three_dimensions=True,
                stroke_width=1.5,
                virtual_time=streak_time,
                opacity=streak_opacity,
                dt=dt,
                color_scheme=scheme,
                min_color_scheme_value=0.0,
                max_color_scheme_value=float(np.quantile(magnitude, 0.995)),
            )
            if colors is not None:
                stream_kwargs["colors"] = colors
            elif streak_color is not None:
                stream_kwargs["color"] = streak_color

            stream_lines = StreamLines(**stream_kwargs)
            stream_lines.start_animation(warm_up=True, flow_speed=1.2)
            self.add(stream_lines)

            if title:
                label = Text(title, font_size=26, color=BLACK).to_corner(UL)
                self.add_fixed_in_frame_mobjects(label)

            self.begin_ambient_camera_rotation(rate=rotation_rate)
            self.wait(run_time)

    return DiracStreamLineScene
