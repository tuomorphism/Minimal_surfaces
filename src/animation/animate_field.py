from manim import *

from ..grid import Grid


def build_vector_field_scene(
    grid: Grid,
    vector_field,
    curve_function: callable = None,
    surface_function: callable = None,
    curve_color=BLACK,
    surface_color=BLACK,
    streak_color=BLUE,
    background_color=WHITE,
    streak_time=0.8,
    streak_opacity=1.0,
    dt=0.05,
    streak_scaling: float = 1.0,
):
    """Build a manim scene showing a 1-form as animated streamlines.

    `vector_field` is a plain (N, N, N, 3) array -- `solution.eta`,
    `solution.eta_0`, or any field on the same grid.
    """
    field = np.asarray(vector_field)

    def map_to_camera_view(point: np.ndarray) -> np.ndarray:
        return (point * 10) - 5

    def inverse_map_to_camera_view(point: np.ndarray) -> np.ndarray:
        return (point + 5) / 10

    def sample(position: np.ndarray) -> np.ndarray:
        idx = tuple(grid.position_to_index(position))
        return field[idx]

    def mapping_of_vector_field(camera_position: np.ndarray) -> np.ndarray:
        mapped = inverse_map_to_camera_view(camera_position)
        mapped = np.clip(mapped, 0.0, grid.length - 1e-9)
        return sample(mapped) * streak_scaling

    def mapping_of_curve(t: float) -> np.ndarray:
        return map_to_camera_view(curve_function(t))

    def mapping_of_surface(u, t) -> np.ndarray:
        return map_to_camera_view(surface_function(u, t))

    class DiracStreamLineScene(ThreeDScene):
        def construct(self):
            self.camera.background_color = background_color
            self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
            self.camera.set_zoom(1.0)

            if curve_function is not None:
                curve = ParametricFunction(
                    mapping_of_curve, t_range=(0, 1), color=curve_color
                )
                self.add(curve)

            if surface_function is not None:
                surface = Surface(
                    mapping_of_surface,
                    u_range=(0, 1),
                    v_range=(0, 1),
                    fill_opacity=0.8,
                    fill_color=surface_color,
                    checkerboard_colors=[surface_color],
                )
                self.add(surface)

            stream_lines = StreamLines(
                func=mapping_of_vector_field,
                x_range=[-4, 4, 1],
                y_range=[-4, 4, 1],
                z_range=[-2, 2, 1],
                stroke_width=1.2,
                virtual_time=streak_time,
                color=streak_color,
                opacity=streak_opacity,
                dt=dt,
            )
            stream_lines.start_animation()
            self.add(stream_lines)
            self.wait(10)

    return DiracStreamLineScene
