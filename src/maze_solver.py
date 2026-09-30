import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AStarMazeSolver:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows = len(maze)
        self.cols = len(maze[0])

    def heuristic(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def get_neighbors(self, node):
        row, col = node

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        neighbors = []

        for dr, dc in directions:
            nr = row + dr
            nc = col + dc

            if (
                0 <= nr < self.rows
                and 0 <= nc < self.cols
                and self.maze[nr][nc] == 0
            ):
                neighbors.append((nr, nc))

        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def solve(self):
        open_set = []

        heapq.heappush(
            open_set,
            (self.heuristic(self.start), 0, self.start)
        )

        came_from = {}
        g_score = {self.start: 0}
        visited = set()

        while open_set:

            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            if current == self.goal:
                return self.reconstruct_path(came_from, current), visited

            for neighbor in self.get_neighbors(current):

                tentative_g = current_g + 1

                if (
                    neighbor not in g_score
                    or tentative_g < g_score[neighbor]
                ):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g
                        + self.heuristic(neighbor)
                    )

                    heapq.heappush(
                        open_set,
                        (f_score, tentative_g, neighbor)
                    )

        return None, visited


def visualize_maze(maze, start, goal, path, visited):

    display_grid = [row[:] for row in maze]

    for cell in visited:
        if cell != start and cell != goal:
            display_grid[cell[0]][cell[1]] = 2

    if path:
        for cell in path:
            if cell != start and cell != goal:
                display_grid[cell[0]][cell[1]] = 3

    display_grid[start[0]][start[1]] = 4
    display_grid[goal[0]][goal[1]] = 5

    cmap = ListedColormap([
        "white",
        "black",
        "lightblue",
        "limegreen",
        "dodgerblue",
        "red"
    ])

    plt.figure(figsize=(10, 7))
    plt.imshow(display_grid, cmap=cmap)

    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))

    plt.grid(True, color="gray", linewidth=0.5)

    plt.title("A* Maze Solver")

    plt.show()


def main():

    # 0 = open cell
    # 1 = wall

    maze = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    goal = (9, 9)

    solver = AStarMazeSolver(
        maze,
        start,
        goal
    )

    path, visited = solver.solve()

    print("\n========== A* MAZE SOLVER ==========\n")
    print(f"Start: {start}")
    print(f"Goal : {goal}")
    print(f"Explored Nodes: {len(visited)}")

    if path:
        print(f"Path Length: {len(path) - 1}")
        print("\nShortest Path:")
        print(path)
    else:
        print("\nNo path found. Goal is unreachable.")

    visualize_maze(
        maze,
        start,
        goal,
        path,
        visited
    )


if __name__ == "__main__":
    main()