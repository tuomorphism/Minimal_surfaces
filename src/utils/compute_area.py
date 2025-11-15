import numpy as np
from src.utils.numerical_methods import numerical_tangent_function

def compute_area_vector(t_values, curve_function):
    h = 1 / t_values.shape[0] # Assumes that curve is parameterized from 0 to 1.
    curve_points = np.array([curve_function(t) for t in t_values])
    tangents = np.array([numerical_tangent_function(t, curve_function, h = h) for t in t_values])
    return np.sum(np.cross(curve_points, tangents), axis = 0) * 0.5 / t_values.shape[0]