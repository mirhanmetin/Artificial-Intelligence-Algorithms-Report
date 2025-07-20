# FIFO Queue Implementation
class MyQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None

    def is_empty(self):
        return len(self.items) == 0

# Timer Replacement
def basic_timer():
    import time
    return time.time()

# 15x15 Grid
grid = [
    [1,1,1,1,1,0,0,0,0,0,0,0,0,1,0],
    [1,0,0,0,1,0,1,1,1,0,1,0,1,1,0],
    [1,0,1,0,1,0,0,0,1,1,1,1,1,0,0],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,1,0],
    [1,0,1,0,1,0,0,0,0,0,1,0,0,0,0],
    [1,0,1,0,1,0,1,1,1,1,1,1,1,1,0],
    [1,0,1,0,1,0,0,0,0,0,1,0,0,0,0],
    [1,0,1,0,0,0,1,1,1,0,1,0,1,1,0],
    [1,0,1,1,1,0,0,0,1,0,1,0,1,0,0],
    [1,0,1,0,0,0,1,0,1,0,0,0,1,1,0],
    [1,0,1,0,1,0,1,0,0,0,1,0,0,1,0],
    [1,0,1,0,1,0,1,0,1,1,1,0,1,1,0],
    [1,0,1,0,1,0,1,0,1,1,1,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,1,1,1],
    [0,0,1,1,1,1,1,1,1,0,1,0,0,0,0]
]

start = (0, 5)
goal = (14, 14)
ROWS = len(grid)
COLS = len(grid[0])

# DFS
def dfs(grid, start, goal):
    stack = [(start, [start])]
    visited = []
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.append(node)
        x, y = node
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and grid[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append(((nx, ny), path + [(nx, ny)]))
    return None

# BFS
def bfs(grid, start, goal):
    queue = MyQueue()
    queue.enqueue((start, [start]))
    visited = []
    while not queue.is_empty():
        node, path = queue.dequeue()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.append(node)
        x, y = node
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and grid[nx][ny] == 0 and (nx, ny) not in visited:
                queue.enqueue(((nx, ny), path + [(nx, ny)]))
    return None

# UCS
def ucs(grid, start, goal):
    queue = [(0, start, [start])]
    visited = []
    while queue:
        queue.sort(key=lambda x: x[0])
        cost, node, path = queue.pop(0)
        if node == goal:
            return path, cost
        if node in visited:
            continue
        visited.append(node)
        x, y = node
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and grid[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((cost+1, (nx, ny), path + [(nx, ny)]))
    return None, float('inf')

# A*
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    queue = [(manhattan(start, goal), 0, start, [start])]
    visited = []
    while queue:
        queue.sort(key=lambda x: x[0])
        f, g, node, path = queue.pop(0)
        if node == goal:
            return path, g
        if node in visited:
            continue
        visited.append(node)
        x, y = node
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and grid[nx][ny] == 0 and (nx, ny) not in visited:
                g_new = g + 1
                f_new = g_new + manhattan((nx, ny), goal)
                queue.append((f_new, g_new, (nx, ny), path + [(nx, ny)]))
    return None, float('inf')

# Main Execution
if __name__ == "__main__":
    print("Final Grid Search Comparison (15x15)\n")

    start_time = basic_timer()
    dfs_path = dfs(grid, start, goal)
    dfs_time = basic_timer() - start_time
    print("DFS Path Length:", len(dfs_path) if dfs_path else "No Path")
    print("DFS Time: {:.6f} seconds\n".format(dfs_time))

    start_time = basic_timer()
    bfs_path = bfs(grid, start, goal)
    bfs_time = basic_timer() - start_time
    print("BFS Path Length:", len(bfs_path) if bfs_path else "No Path")
    print("BFS Time: {:.6f} seconds\n".format(bfs_time))

    start_time = basic_timer()
    ucs_path, ucs_cost = ucs(grid, start, goal)
    ucs_time = basic_timer() - start_time
    print("UCS Path Length:", len(ucs_path) if ucs_path else "No Path")
    print("UCS Cost:", ucs_cost)
    print("UCS Time: {:.6f} seconds\n".format(ucs_time))

    start_time = basic_timer()
    astar_path, astar_cost = astar(grid, start, goal)
    astar_time = basic_timer() - start_time
    print("A* Path Length:", len(astar_path) if astar_path else "No Path")
    print("A* Cost:", astar_cost)
    print("A* Time: {:.6f} seconds\n".format(astar_time))
