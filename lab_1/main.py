from animation import Animation
import numpy as np
import sympy as sp
from sympy import sin, cos


class MatPoint:
    def __init__(self, pos, symbol):
        self.__t = symbol
        self.__position = pos
        self.__velocity = [sp.diff(component, self.__t, 1) for component in pos]
        self.__acceleration = [sp.diff(component, self.__t, 2) for component in pos]

    # Dependency of the point's coordinates on time
    def position(self, time: float) -> np.ndarray:
        return np.array([float(comp.subs(self.__t, time).evalf()) for comp in self.__position])

    # Dependency of the point's velocity on time
    def velocity(self, time: float) -> np.ndarray:
        return np.array([float(comp.subs(self.__t, time).evalf()) for comp in self.__velocity])

    # Dependency of the point's acceleration on time
    def acceleration(self, time: float) -> np.ndarray:
        return np.array([float(comp.subs(self.__t, time).evalf()) for comp in self.__acceleration])

    # Tangential vector at the current time
    def tangential(self, time: float) -> np.ndarray:
        vel = self.velocity(time)
        return vel / np.linalg.norm(vel)

    # Normal vector at the current time
    def normal(self, time: float) -> np.ndarray:
        acc = self.acceleration(time)
        return acc / np.linalg.norm(acc)

    # Binormal vector at the current time
    def binormal(self, time: float) -> np.ndarray:
        tan = self.tangential(time)
        norm = self.normal(time)
        return np.cross(tan, norm)



def main() -> None:
    t = sp.symbols('t')

    # ----- parametric equations ---------
    position = [
        sin(t)*cos(1-t),
        sin(t)*cos(t),
        cos(1-t)
    ]
    # ------------------------------------

    animation = Animation()
    mat_point = MatPoint(
        pos=position,
        symbol=t,
    )

    # Initial and final moments of time
    start = 0
    end = 50

    # Moving point
    animation.add_point(update=mat_point.position, color='b', label='Material point', trail_length=1000)

    # Position vector
    animation.add_vector(start=None, update=mat_point.position, color='r', label='Position vector')

    # Velocity vector
    animation.add_vector(mat_point.position, mat_point.velocity, 'orange', 'Velocity vector')

    # Acceleration vector
    animation.add_vector(mat_point.position, mat_point.acceleration, 'g', 'Acceleration vector')

    # Tangential, normal, and binormal vectors
    animation.add_vector(mat_point.position, mat_point.normal, 'gray', label='Normal vector')
    animation.add_vector(mat_point.position, mat_point.binormal, 'b', label='Binormal vector')

    animation.start_animation(start, end, 600, speed=15)


if __name__ == '__main__':
    main()
