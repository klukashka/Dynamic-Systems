import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing import Callable


class Animation:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.graph_objects = {
            'points': [],
            'vectors': [],
            'trails': []
        }

        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def add_point(self, update, color, label: str, trail_length=50):
        # Initialize the graphical object for the point
        point_graph, = self.ax.plot([], [], [], color + 'o', label=label)
        self.graph_objects['points'].append((point_graph, update))

        # Initialize the trail (line)
        trail_graph, = self.ax.plot([], [], [], color=color, lw=1)  # Trail line
        self.graph_objects['trails'].append((trail_graph, [], trail_length))  # Add trail

    def add_vector(
            self,
            start: Callable[[float], np.ndarray] | None,
            update: Callable[[float], np.ndarray],
            color: str,
            label: str = None
    ) -> None:
        if start is None:
            def start(_): return np.zeros((3,))
        vector_graph = self.ax.quiver(0, 0, 0, 0, 0, 0, color=color, label=label)
        self.graph_objects['vectors'].append((vector_graph, start, color, label, update))

    def start_animation(self, start: float, end: float, number_of_frames: int, speed: float = 1.0) -> None:
        def update(frame):
            ret = []
            # Update points
            for i, (point_graph, upd) in enumerate(self.graph_objects['points']):
                pos = upd(frame)
                point_graph.set_data([pos[0]], [pos[1]])  # Update x and y
                point_graph.set_3d_properties([pos[2]])   # Update z
                ret.append(point_graph)

                # Update trail
                trail_graph, positions, trail_length = self.graph_objects['trails'][i]
                positions.append(pos)  # Save the current position
                if len(positions) > trail_length:
                    positions.pop(0)  # Limit the trail length

                # Update trail data
                trail_positions = np.array(positions)
                trail_graph.set_data(trail_positions[:, 0], trail_positions[:, 1])
                trail_graph.set_3d_properties(trail_positions[:, 2])
                ret.append(trail_graph)

            # Update vectors
            for i, (vector_graph, start, color, label, upd) in enumerate(self.graph_objects['vectors']):
                vector_graph.remove()
                start_pos = start(frame)
                direction = upd(frame)
                vector_graph = self.ax.quiver(*start_pos, *direction, color=color, label=label)
                self.graph_objects['vectors'][i] = (vector_graph, start, color, label, upd)
                ret.append(vector_graph)

            return ret

        frames = np.linspace(start, end, number_of_frames)
        ani = FuncAnimation(self.fig, update, frames=frames, init_func=lambda: update(start), blit=False,
                            interval=1000 * (end - start) / (number_of_frames * speed))

        plt.legend()
        plt.show()
