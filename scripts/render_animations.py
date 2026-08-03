"""Render the manim animations used by minimal_surfaces.ipynb into assets/.

Separate from `render_assets.py` because manim is an optional dependency with
system-library requirements (cairo, pango, ffmpeg):

    pip install -e ".[animation]"
    python scripts/render_animations.py               # all three
    python scripts/render_animations.py optimization  # one by name

Each scene is a few minutes. Outputs land in assets/ as .mp4, re-encoded down
from manim's very generous default bitrate.

The notebook embeds these with <video> tags, which render in Jupyter and VS Code
but not on GitHub; the static montage in assets/admm_iterations.png covers that
case, so no GIF duplicates are kept. `mp4_to_gif` is here if you want one.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import *  # noqa: E402,F401,F403
from skimage import measure  # noqa: E402

from src import curves, extract, spectral  # noqa: E402
from src.animation.animate_field import (  # noqa: E402
    BACKGROUND,
    CURVE_COLOR,
    build_vector_field_scene,
    curve_mobject,
    mesh_mobject,
)
from src.grid import Grid  # noqa: E402
from src.initial_guess import compute_initial_guess  # noqa: E402
from src.plateau import solve_plateau  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
RADIUS = 0.3
QUALITY = "medium_quality"  # 720p30


def circle(t):
    return curves.circle(t, radius=RADIUS)


def render(scene_cls, name, quality=QUALITY, fmt="mp4"):
    """Render a scene and move the result into assets/<name>.<fmt>."""
    with tempfile.TemporaryDirectory() as tmp:
        with tempconfig(
            {
                "quality": quality,
                "format": fmt,
                "media_dir": tmp,
                "output_file": name,
                "disable_caching": True,
                "background_color": BACKGROUND,
                "verbosity": "WARNING",
            }
        ):
            scene = scene_cls()
            scene.render()
            produced = Path(scene.renderer.file_writer.movie_file_path)
            target = ASSETS / f"{name}.{fmt}"
            if fmt == "mp4":
                _compress(produced, target)
            else:
                shutil.copy(produced, target)
    print(f"  wrote {target.relative_to(ASSETS.parent)} "
          f"({target.stat().st_size // 1024} KB)")
    return target


def _compress(source, target, crf=30):
    """Re-encode with x264.

    Manim's own encode is generous -- a 12 s streamline scene lands around 19 MB,
    which is not something to commit to a repository. Re-encoding at crf 30 with
    faststart cuts that by an order of magnitude with no visible loss on this
    kind of content.
    """
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        check=True,
    )


def mp4_to_gif(mp4_path, width=520, fps=12):
    """Downscaled GIF companion, for viewers that will not play a <video> tag.

    Two-pass with a generated palette; a straight conversion dithers badly on
    the smooth colour ramps these scenes are full of.
    """
    import subprocess

    gif_path = mp4_path.with_suffix(".gif")
    palette = mp4_path.with_name(mp4_path.stem + "_palette.png")
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path),
         "-vf", f"{scale},palettegen=stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path), "-i", str(palette),
         "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
         str(gif_path)],
        check=True,
    )
    palette.unlink(missing_ok=True)
    print(f"  wrote {gif_path.relative_to(ASSETS.parent)} "
          f"({gif_path.stat().st_size // 1024} KB)")
    return gif_path


def isosurface(magnitude, level):
    """Marching cubes on a scalar grid, returned in unit-cube coordinates."""
    verts, faces, _, _ = measure.marching_cubes(magnitude, level=level)
    return verts / magnitude.shape[0], faces


# ------------------------------------------------------------------ scenes


def anim_initial_guess():
    """The Biot-Savart field of the boundary curve, as flowing field lines.

    This is literally the magnetic field of a current loop, which is what makes
    eta_0 a natural feasible starting point: its curl is exactly delta_Gamma.
    """
    grid = Grid(48)
    guess = compute_initial_guess(grid, circle)

    scene_cls = build_vector_field_scene(
        grid,
        guess.eta_0,
        curve_function=circle,
        colors=[ManimColor("#2b2f77"), ManimColor("#3b8fc0"), ManimColor("#f0b73f")],
        streak_scaling=1.6,
        normalize_field=True,
        streak_time=2.5,
        dt=0.04,
        rotation_rate=0.20,
        run_time=9.0,
        seed_spacing=0.95,
        title="eta_0 : the Biot-Savart field of the curve",
    )
    render(scene_cls, "anim_initial_guess")


def anim_optimization(resolution=32):
    """The core of the method: |eta| collapsing onto the minimal surface.

    The *extracted mesh* would be a poor choice here -- the boundary constraint
    holds from iteration 1, so the level set already encodes a disc throughout
    and nothing visible would change. What actually evolves is the support of
    eta: it starts as a fat diffuse lens and the shrinkage squeezes it onto the
    surface, with the peak magnitude climbing as the mass concentrates.
    """
    gamma = circle
    wanted = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 130]
    snapshots = {}

    def callback(k, X, phi, history):
        if k in wanted:
            snapshots[k] = (spectral.pointwise_norm(X).copy(), history["mass"][-1])

    print("  solving with snapshots ...", flush=True)
    solution = solve_plateau(gamma, resolution=resolution, rtol=1e-3, max_iter=400,
                             callback=callback)
    snapshots[solution.iterations] = (
        spectral.pointwise_norm(solution.eta), solution.mass
    )

    frames = []
    for k in sorted(snapshots):
        magnitude, mass = snapshots[k]
        verts, faces = isosurface(magnitude, 0.3 * magnitude.max())
        frames.append((k, verts, faces, mass, float(magnitude.max())))
        print(f"    iter {k:4d}: {len(faces):5d} faces, mass {mass:.4f}", flush=True)

    exact = np.pi * RADIUS**2
    boundary = curve_mobject(gamma)

    class OptimizationScene(ThreeDScene):
        def construct(self):
            self.camera.background_color = BACKGROUND
            self.set_camera_orientation(phi=72 * DEGREES, theta=-50 * DEGREES, zoom=1.9)
            self.add(boundary)

            caption = Text("", font_size=24, color=BLACK).to_corner(UL)
            self.add_fixed_in_frame_mobjects(caption)
            self.begin_ambient_camera_rotation(rate=0.16)

            current = None
            for i, (k, verts, faces, mass, peak) in enumerate(frames):
                mesh = mesh_mobject(verts, faces, cmap="magma", opacity=0.75)
                text = Text(
                    f"iteration {k}          mass {mass:.4f}   (exact {exact:.4f})",
                    font_size=24,
                    color=BLACK,
                ).to_corner(UL)
                self.remove(caption)
                caption = text
                self.add_fixed_in_frame_mobjects(caption)

                if current is None:
                    self.play(FadeIn(mesh), run_time=0.6)
                else:
                    self.play(FadeOut(current), FadeIn(mesh), run_time=0.45)
                current = mesh
                self.wait(0.35 if i < len(frames) - 1 else 2.2)

    render(OptimizationScene, "anim_optimization")


def anim_gallery():
    """A turntable of finished surfaces, including a multi-component boundary."""
    entries = [
        ("trefoil knot", curves.trefoil, dict()),
        ("Borromean rings", curves.borromean_rings(), dict(clip_fraction=0.22)),
    ]
    built = []
    for title, gamma, kw in entries:
        print(f"  solving {title} ...", flush=True)
        solution = solve_plateau(gamma, resolution=48, rtol=1e-3, max_iter=400)
        verts, faces = extract.extract_surface(solution, **kw)
        built.append((title, gamma, verts, faces, solution.mass))

    class GalleryScene(ThreeDScene):
        def construct(self):
            self.camera.background_color = BACKGROUND
            self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, zoom=1.7)
            self.begin_ambient_camera_rotation(rate=0.35)

            caption = Text("", font_size=24, color=BLACK).to_corner(UL)
            self.add_fixed_in_frame_mobjects(caption)
            previous = None

            for title, gamma, verts, faces, mass in built:
                group = VGroup(
                    mesh_mobject(verts, faces, cmap="viridis", opacity=0.95),
                    curve_mobject(gamma),
                )
                text = Text(f"{title}    mass {mass:.4f}", font_size=24, color=BLACK)
                text.to_corner(UL)
                self.remove(caption)
                caption = text
                self.add_fixed_in_frame_mobjects(caption)

                if previous is None:
                    self.play(FadeIn(group), run_time=0.8)
                else:
                    self.play(FadeOut(previous), FadeIn(group), run_time=0.8)
                previous = group
                self.wait(5.0)

    render(GalleryScene, "anim_gallery")


SCENES = {
    "initial_guess": anim_initial_guess,
    "optimization": anim_optimization,
    "gallery": anim_gallery,
}


def main(names):
    ASSETS.mkdir(exist_ok=True)
    for name in names or SCENES:
        print(f"[{name}]", flush=True)
        SCENES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
