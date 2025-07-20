BOX_ROWS = 2
BOX_COLS = 3
N = 6

# Grid with 14 blanks (0 = empty cell)
grid = [
    [5, 3, 0, 1, 0, 2],
    [2, 0, 4, 0, 3, 0],
    [1, 0, 0, 3, 6, 5],
    [0, 5, 3, 0, 0, 1],
    [3, 2, 0, 0, 1, 4],
    [0, 6, 1, 0, 5, 0]
]

def is_valid(grid, row, col, num):
    # Row check
    if num in grid[row]:
        return False
    # Column check
    for i in range(N):
        if grid[i][col] == num:
            return False
    # Box check
    box_row_start = (row // BOX_ROWS) * BOX_ROWS
    box_col_start = (col // BOX_COLS) * BOX_COLS
    for i in range(box_row_start, box_row_start + BOX_ROWS):
        for j in range(box_col_start, box_col_start + BOX_COLS):
            if grid[i][j] == num:
                return False
    return True

def find_empty(grid):
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                return (i, j)
    return None

def solve(grid):
    empty = find_empty(grid)
    if not empty:
        return True  # no empty cells = solved
    row, col = empty
    for num in range(1, N + 1):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            if solve(grid):
                return True
            grid[row][col] = 0  # backtrack
    return False

def print_grid(grid):
    for row in grid:
        print(" ".join(str(num) for num in row))

# Run solver
print("Original Grid with Blanks (0):")
print_grid(grid)
print("\nSolving...\n")

if solve(grid):
    print("Solved Sudoku Grid:")
    print_grid(grid)
else:
    print("No solution found.")
