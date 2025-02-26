import numpy as np
from typing import List, Tuple

def bilinear_interpolation(x: float,
                           y: float,
                           known_points: List[Tuple[float,float]],
                           values_of_function: List[float]) -> float:
    """
    Perform bilinear interpolation for a given (x, y) based on four known points and their corresponding function values.
    
    :param x: The x-coordinate where interpolation is performed.
    :param y: The y-coordinate where interpolation is performed.
    :param known_points: List or tuple of the four known corner points [(x1, y1), (x2, y1), (x1, y2), (x2, y2)].
    :param values_of_function: List or tuple of function values at the four corner points 
                               [f(x1, y1), f(x1, y2), f(x2, y1), f(x2, y2)].
    :return: The interpolated function value at (x, y).
    """

    x_1, y_1, x_2, y_2 = known_points
    f_q11, f_q12, f_q21, f_q22 = values_of_function
    delta_x = np.array([x_2 - x, x - x_1])
    delta_y = np.array([y_2 - y, y - y_1])  
    func_val_matrix = np.array([[f_q11, f_q12], [f_q21, f_q22]])
    interpolated_value = (delta_x @ (func_val_matrix @ delta_y)) / ((x_2 - x_1) * (y_2 - y_1))
    return interpolated_value

if name == "__main__" : 
    known_points = [(1, 2), (3, 2), (1, 4), (3, 4)]
    values_of_function = [5, 6, 7, 8]
    x, y = 2, 3
    result = bilinear_interpolation(x, y, known_points, values_of_function)
    print(f"Interpolated value at ({x}, {y}) using bilinear interpolation: {result}")
