import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QGraphicsScene, QGraphicsView
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import heapq
import random

class SearchAlgorithmGUI(QMainWindow):
    def __init__(self):
        super(SearchAlgorithmGUI, self).__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.start_label = QLabel('Start:', self)
        self.end_label = QLabel('End:', self)
        self.run_button = QPushButton('Run Dijkstra', self)
        self.run_button.clicked.connect(self.run_dijkstra)

        self.run_buttonA = QPushButton('Run A*', self)
        self.run_buttonA.clicked.connect(self.run_a_star)

        self.run_buttonB = QPushButton('Run BFS', self)
        self.run_buttonB.clicked.connect(self.run_bfs)

        self.run_buttonD = QPushButton('Run DFS', self)
        self.run_buttonD.clicked.connect(self.run_dfs)

        self.show_map_button = QPushButton('Show Map', self)
        self.show_map_button.clicked.connect(self.show_map)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.start_label)
        self.layout.addWidget(self.end_label)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.run_buttonA)
        self.layout.addWidget(self.run_buttonB)
        self.layout.addWidget(self.run_buttonD)
        self.layout.addWidget(self.show_map_button)
        self.layout.addWidget(self.canvas)

        self.box_size = 20  # Increase box size to accommodate more obstacles
        self.obstacle_density = 0.2  # Density of obstacles in the square
        self.path = []  # Initialize the path attribute

        self.map_dialog = None  # Initialize the map_dialog attribute

    def run_bfs(self):
        start, end = self.get_valid_start_end()
        iterations, progress = self.bfs_algorithm(start, end)
        self.plot_graph(iterations, progress)

    def bfs_algorithm(self, start, end):
        queue = [(0, start, [])]  # Updated queue to include the path taken
        visited = set()
        iterations = []
        progress = []

        while queue:
            (cost, current, path) = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    queue.append((cost + 1, neighbor, path + [current]))

        return iterations, progress



    def run_a_star(self):
        start, end = self.get_valid_start_end()
        iterations, progress = self.a_star_algorithm(start, end)
        self.plot_graph(iterations, progress)

    def reconstruct_path(self, start, end, g_scores, max_iterations=1000):
        path = set()
        current = end
        iterations = 0

        while current != start and iterations < max_iterations:
            path.add(current)
            neighbors = self.get_neighbors(current)

            # Find the unvisited neighbor with the lowest cost (g_score)
            current = min(neighbors, key=lambda n: g_scores.get(n, float('inf')))
            iterations += 1

        path.add(start)
        return path

    def a_star_algorithm(self, start, end):
        open_set = {start}
        closed_set = set()
        g_scores = {start: 0}
        f_scores = {start: self.distance(start, end)}
        iterations = []
        progress = []

        while open_set:
            current = min(open_set, key=lambda node: f_scores[node])

            if current == end:
                path = self.reconstruct_path(start, end, g_scores.copy())  # Passed a copy of g_scores
                self.path = path
                return iterations, progress

            open_set.remove(current)
            closed_set.add(current)

            iterations.append(len(closed_set))
            progress.append(g_scores.get(current, 0))  # Use get to avoid KeyError

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed_set or neighbor in self.get_obstacles():
                    continue

                tentative_g_score = g_scores.get(current, 0) + 1  # Use get to avoid KeyError

                if neighbor not in open_set or tentative_g_score < g_scores.get(neighbor, 0):  # Use get to avoid KeyError
                    open_set.add(neighbor)
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.distance(neighbor, end)

        return iterations, progress

    


    def run_dijkstra(self):
        start, end = self.get_valid_start_end()
        iterations, progress = self.dijkstra_algorithm(start, end)
        self.plot_graph(iterations, progress)

    def dijkstra_algorithm(self, start, end):
        visited = set()
        heap = [(0, start, [])]  # Updated heap to include the path taken
        iterations = []
        progress = []

        while heap:
            (cost, current, path) = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    heapq.heappush(heap, (cost + 1, neighbor, path + [current]))

        return iterations, progress

    def run_bfs(self):
        start, end = self.get_valid_start_end()
        iterations, progress = self.bfs_algorithm(start, end)
        self.plot_graph(iterations, progress)

    def bfs_algorithm(self, start, end):
        queue = [(0, start, [])]  # Updated queue to include the path taken
        visited = set()
        iterations = []
        progress = []

        while queue:
            (cost, current, path) = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    queue.append((cost + 1, neighbor, path + [current]))

        return iterations, progress
    
    def run_dfs(self):
        start, end = self.get_valid_start_end()
        iterations, progress = self.dfs_algorithm(start, end)
        self.plot_graph(iterations, progress)

    def dfs_algorithm(self, start, end):
        stack = [(0, start, [])]  # Updated stack to include the path taken
        visited = set()
        iterations = []
        progress = []

        while stack:
            (cost, current, path) = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    stack.append((cost + 1, neighbor, path + [current]))

        return iterations, progress
   
    def get_neighbors(self, current):
        x, y = current
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(x, y) for (x, y) in neighbors if 0 <= x < self.box_size and 0 <= y < self.box_size and (x, y) not in self.get_obstacles()]


    def get_obstacles(self):
    # Define your predefined obstacles as a list of coordinates
        predefined_obstacles = [(1, 2), (0, 2), (3, 2), (4, 2), (3, 2), (6, 2), (7, 2),
                                (7, 3), (0, 4), (7, 5), (7, 6), (0, 6), (5, 9), (4, 6),
                                (3, 6), (2, 6), (2, 8), (2, 4), (2, 3), (0,3), (12,9), 
                                (2,17),(0,16),(1,19),(3,17),(12,19),(5,0),(7,1),
                                (14,11),(11,14),(12,12),(5,17),(6,15),(9,11),
                                (15,8), (16, 6), (12,18)]

        return predefined_obstacles

    def get_valid_start_end(self):
        start = (0, 0)  # Top left corner
        end = (self.box_size - 1, self.box_size - 1)  # Bottom right corner
        return start, end

    def get_shortest_path(self, start, end):
        path = set()
        current = end
        while current != start:
            path.add(current)
            neighbors = self.get_neighbors(current)
            current = min(neighbors, key=lambda n: self.distance(n, start))
        path.add(start)
        return path

    def distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plot_graph(self, iterations, progress):
        self.ax.clear()
        self.ax.plot(iterations, progress, label='Progress')
        self.ax.legend()
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Progress (Cost)')
        self.canvas.draw()

    def show_map(self):
        if self.map_dialog is None:
            self.map_dialog = MapDialog(self)
        self.map_dialog.update_map(self.box_size, self.get_obstacles(), self.path)
        self.map_dialog.show()


class MapDialog(QWidget):
    def __init__(self, parent=None):
        super(MapDialog, self).__init__(parent)
        self.setWindowTitle('Map View')
        self.setGeometry(200, 200, 400, 400)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.close_button = QPushButton('Close', self)
        self.close_button.clicked.connect(self.close_dialog)

        
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.close_button)
        self.layout.addWidget(self.view)
        
        

    def update_map(self, box_size, obstacles, path):
        self.scene.clear()

        # Draw square box
        self.scene.addRect(0, 0, box_size * 20, box_size * 20, pen=Qt.black)

        # Draw obstacles
        for obstacle in obstacles:
            x, y = obstacle
            self.scene.addRect(x * 20, y * 20, 20, 20, brush=Qt.gray)

        # Draw path
        for node in path:
            x, y = node
            self.scene.addEllipse(x * 20, y * 20, 20, 20, pen=Qt.green)

        # Show view
        self.view.setSceneRect(0, 0, box_size * 20, box_size * 20)
        self.view.setScene(self.scene)
    
    def close_dialog(self):
        self.close()

def main():
    app = QApplication(sys.argv)
    window = SearchAlgorithmGUI()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('Search Algorithm GUI')
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()