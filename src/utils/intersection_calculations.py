import numpy as np
from src.utils.levi_civita import generate_symbols

def SignedIntersection(Γ, f, return_crossings = False):
    """
    Compute the flux-weighted signed intersection of a discrete curve segment with a finite rectangular grid face,
    ensuring that the final sum is properly normalized.

    Parameters:
    Γ (list of tuples): List of curve points [(x1, y1, z1), (x2, y2, z2), ...].
    f (tuple): (v, e_i, size) where:
        - v (tuple): Base vertex defining the face (x, y, z).
        - e_i (int): Normal direction index (0=x,1=y,2=z).
        - size (tuple): Size of the face in the two perpendicular directions.

    Returns:
    (tuple) containing
    float: Net flux-weighted signed intersection value, constrained to [-1,1].
    int: The number of crossings
    """
    v, e_i, size = f  # Extract face properties
    normal_axis = e_i  # The axis normal to the face
    
    # Define the normal vector of the face
    face_normal = np.zeros(3)
    face_normal[normal_axis] = 1  # Example: If normal is in the Z direction, it's (0,0,1)

    # Levi-Civita symbol should equal 1 for basis indices j, k for basis vector e_i.
    total_permutations = generate_symbols()
    _i, j, k = total_permutations[e_i][0]
    
    # Compute face boundaries based on size
    face_min = np.array([v[j], v[k]])  # Min corner in (j,k) plane
    face_max = face_min + np.array(size)  # Max corner in (j,k) plane
    
    total_flux = 0.0  # Accumulate weighted signed contributions
    crossing_count = 0  # Track number of valid crossings

    for segment_index in range(len(Γ) - 1):
        p1 = np.array(Γ[segment_index])
        p2 = np.array(Γ[segment_index + 1])

        # Extract coordinate values along the normal direction
        coord1, coord2 = p1[normal_axis], p2[normal_axis]
        face_coord = v[normal_axis]  # The face coordinate in this direction

        # Check if the segment crosses the plane of the face
        if (coord1 < face_coord and coord2 > face_coord) or (coord1 > face_coord and coord2 < face_coord):
            
            # Compute intersection point
            t = (face_coord - coord1) / (coord2 - coord1)  
            intersection = p1 + t * (p2 - p1)  
            intersection_2d = np.array([intersection[j], intersection[k]])

            # Check if the intersection lies within the finite face
            if np.all(face_min <= intersection_2d) and np.all(intersection_2d <= face_max):
                
                # Compute curve tangent at intersection
                curve_tangent = (p2 - p1) / np.linalg.norm(p2 - p1) 

                # Compute projection of tangent onto face normal
                flux_contribution = np.dot(curve_tangent, face_normal)
                
                total_flux += flux_contribution
                crossing_count += 1
    
    if return_crossings:
        return (total_flux, crossing_count)
    return total_flux
