from typing import Callable, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay


def chebyshev_polynomial(n: int, x: float) -> float:
    """
    Вычисление полинома Чебышёва степени n в точке x.
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev_polynomial(n - 1, x) - chebyshev_polynomial(n - 2, x)

def default_basis(x: float, y: float, i: int, j: int) -> float:
    """
    Дефолтная базисная функция на основе полиномов Чебышёва.
    :param x, y: Точки, в которых вычисляется базисная функция.
    :param i, j: Степени полиномов Чебышёва по x и y.
    :return: Значение базисной функции в точке (x, y).
    """
    return chebyshev_polynomial(i, x) * chebyshev_polynomial(j, y)

def compute_mass_matrix(points: List[Tuple[float, float]], 
                        triangles: List[Tuple[int, int, int]],
                        basis_func: Callable, 
                        max_degree: int) -> csr_matrix:
    """
    Вычисление матрицы масс для треугольной сетки с использованием базиса Чебышёва.
    
    :param points: Список точек сетки.
    :param triangles: Список треугольников.
    :param basis_func: Функция базиса.
    :param max_degree: Максимальная степень полиномов Чебышёва.
    :return: Матрица масс (разреженная).
    """
    n_basis = (max_degree + 1) ** 2
    mass_matrix = np.zeros((n_basis, n_basis))

    for triangle in triangles:
        vertices = [points[i] for i in triangle]
        area = 0.5 * abs(
            (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1]) -
            (vertices[2][0] - vertices[0][0]) * (vertices[1][1] - vertices[0][1])
        )
        local_mass = np.zeros((n_basis, n_basis))
        for i in range(max_degree + 1):
            for j in range(max_degree + 1):
                for k in range(max_degree + 1):
                    for l in range(max_degree + 1):
                        x_center = (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3
                        y_center = (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3
                        local_mass[i * (max_degree + 1) + j, k * (max_degree + 1) + l] = area * basis_func(x_center, y_center, i, j) * basis_func(x_center, y_center, k, l)
        # Добавляем локальную матрицу масс к глобальной
        for i in range(n_basis):
            for j in range(n_basis):
                mass_matrix[i, j] += local_mass[i, j]

    return csr_matrix(mass_matrix)

def compute_load_vector(points: List[Tuple[float, float]], 
                        triangles: List[Tuple[int, int, int]], 
                        source_values: List[float],
                        basis_func: Callable,
                        max_degree: int) -> np.ndarray:
    """
    Вычисление вектора нагрузки для метода Галеркина.
    
    :param points: Список точек сетки.
    :param triangles: Список треугольников.
    :param source_values: Значения функции на исходной сетке.
    :param basis_func: Функция базиса.
    :param max_degree: Максимальная степень полиномов Чебышёва.
    :return: Вектор нагрузки.
    """
    n_basis = (max_degree + 1) ** 2
    load_vector = np.zeros(n_basis)

    for triangle in triangles:
        vertices = [points[i] for i in triangle]
        area = 0.5 * abs(
            (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1]) -
            (vertices[2][0] - vertices[0][0]) * (vertices[1][1] - vertices[0][1])
        )
        local_load = np.zeros(n_basis)
        for i in range(max_degree + 1):
            for j in range(max_degree + 1):
                x_center = (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3
                y_center = (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3
                local_load[i * (max_degree + 1) + j] = area * source_values[triangle[0]] * basis_func(x_center, y_center, i, j)
        load_vector += local_load

    return load_vector

def galerkin_interpolation(points: List[Tuple[float, float]],
                           triangles: List[Tuple[int, int, int]],
                           source_values: List[float], 
                           basis_func: Callable, 
                           max_degree: int) -> List[float]:
    """
    Интерполяция методом Галеркина с использованием базиса Чебышёва.
    
    :param points: Список точек сетки.
    :param triangles: Список треугольников.
    :param source_values: Значения функции на исходной сетке.
    :param basis_func: Функция базиса.
    :param max_degree: Максимальная степень полиномов Чебышёва.
    :return: Интерполированные значения на сетке.
    """
    # Вычисляем матрицу масс и вектор нагрузки
    mass_matrix = compute_mass_matrix(points, triangles, basis_func, max_degree)
    load_vector = compute_load_vector(points, triangles, source_values, basis_func, max_degree)

    # Решаем систему уравнений
    interpolation_weights = spsolve(mass_matrix, load_vector)

    # Интерполяция на всех точках сетки
    interpolated_values = []
    for point in points:
        x, y = point
        interpolated_value = 0.0
        for i in range(max_degree + 1):
            for j in range(max_degree + 1):
                interpolated_value += interpolation_weights[i * (max_degree + 1) + j] * basis_func(x, y, i, j)
        interpolated_values.append(interpolated_value)

    return interpolated_values

def generate_grid(n_points: int,
                  x_min: float,
                  x_max: float, 
                  y_min: float,
                  y_max: float) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int]]]:
    """
    Генерация треугольной сетки.
    
    :param n_points: Количество точек.
    :param x_min, x_max: Границы по оси x.
    :param y_min, y_max: Границы по оси y.
    :return: Список точек и список треугольников.
    """
    # Генерация случайных точек
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_points, 2))
    points = [tuple(point) for point in points]
    # Триангуляция Делоне
    tri = Delaunay(points)
    triangles = [tuple(triangle) for triangle in tri.simplices]
    return points, triangles

def plot_grid(points: List[Tuple[float, float]], triangles: List[Tuple[int, int, int]], source_values: List[float], interpolated_values: List[float]):
    """
    Визуализация исходной сетки и результата интерполяции.
    
    :param points: Список точек сетки.
    :param triangles: Список треугольников.
    :param source_values: Значения функции на исходной сетке.
    :param interpolated_values: Интерполированные значения.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.tricontourf([point[0] for point in points], [point[1] for point in points], triangles, source_values, levels=20, cmap='plasma')
    plt.title("Исходная сетка")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.tricontourf([point[0] for point in points], [point[1] for point in points], triangles, interpolated_values, levels=20, cmap='plasma')
    plt.title("Результат интерполяции")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

@click.command()
@click.option("--n_points",
              default=20,
              type=int,
              help="Number of points in the grid"
)
@click.option("--x_min",
              default=0.0,
              type=float,
              help="Minimum x-coordinate of the grid"
)
@click.option("--x_max",
              default=1.0,
              type=float,
              help="Maximum x-coordinate of the grid"
)
@click.option("--y_min",
              default=0.0,
              type=float,
              help="Minimum y-coordinate of the grid"
)
@click.option("--y_max",
              default=1.0,
              type=float,
              help="Maximum y-coordinate of the grid"
)
@click.option("--max_degree",
              default=3,
              type=int,
              help="Maximum degree of Chebyshev polynomials"
)
@click.option("--basis_func",
              default="default",
              type=str,
              help="Basis function to use (default: Chebyshev)"
)
def main(n_points: int,
         x_min: float,
         x_max: float,
         y_min: float,
         y_max: float,
         max_degree: int,
         basis_func: str) -> None:
    # Генерация сетки
    points, triangles = generate_grid(n_points, x_min, x_max, y_min, y_max)
    
    # Задание функции на исходной сетке
    def source_function(x, y):
        return np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)
    source_values = [source_function(point[0], point[1]) for point in points]
    
    # Выбор базисной функции
    if basis_func == "default":
        basis_func = default_basis
    else:
        def custom_basis(x, y, i, j):
            return np.sin(i * x) * np.cos(j * y)
        basis_func = custom_basis

    interpolated_values = galerkin_interpolation(points, triangles, source_values, basis_func, max_degree)
    plot_grid(points, triangles, source_values, interpolated_values)
    return
    
if __name__ == "__main__":
    main()