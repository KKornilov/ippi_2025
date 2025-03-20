# Interpolation Techniques

This project demonstrates various interpolation techniques, including bilinear interpolation and L2-norm (Galerkin) interpolation. Below is a detailed description of each method and the corresponding code.

---

## Bilinear Interpolation (`bilinear_interpolation.py`)

### Description
Bilinear interpolation is a classical method used to interpolate functions of two variables (e.g., \( f(x, y) \)) on a 2D grid. It works by linearly interpolating first in one direction (e.g., along the x-axis) and then in the other direction (e.g., along the y-axis).

### Input Parameters
- **`x`**: The x-coordinate of the point where you want to interpolate the function.
- **`y`**: The y-coordinate of the point where you want to interpolate the function.
- **`coords`**: The coordinates of the known points in the format $[x_1, x_2, y_1, y_2]$.
- **`values`**: The function values at the known points in the format $[f(x_1, y_1), f(x_1, y_2), f(x_2, y_1), f(x_2, y_2)]$.

--- 

L2-Norm (Galerkin) Interpolation (l2_interpolation.py)

## Description

L2-norm interpolation, also known as Galerkin interpolation, is a method that minimizes the L2-norm of the error between the interpolated function and the true function. This method is particularly useful when dealing with irregular grids or when higher accuracy is required. It uses a basis of functions (e.g., Chebyshev polynomials) to approximate the solution.

## Input Parameters

- **`point`s** : A list of tuples representing the coordinates of the grid points.
- **`triangles`**: A list of tuples representing the triangles formed by the grid points (obtained via Delaunay triangulation).
- **`source_values`**: The function values at the known grid points.
- **`basis_func`**: The basis function used for interpolation (default is Chebyshev polynomials).
- **`max_degree`**: The maximum degree of the polynomials used in the basis function.