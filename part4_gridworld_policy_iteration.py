import numpy as np

# GridWorld dimensions and reward setup
rows, cols = 3, 4
grid = np.zeros((rows, cols))
grid[0][3] = 1   # win state
grid[1][3] = -1  # loss state

# MDP parameters
gamma = 0.9
threshold = 1e-4

actions = ['U', 'D', 'L', 'R']
action_vectors = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# Transition probabilities
P_FORWARD = 0.8
P_LEFT = 0.1
P_RIGHT = 0.1

def in_bounds(x, y):
    return 0 <= x < rows and 0 <= y < cols

def is_terminal(state):
    return state in [(0, 3), (1, 3)]

def get_transitions(state, action):
    i, j = state
    dirs = list(action_vectors.keys())
    forward = action
    left = dirs[(dirs.index(action) - 1) % 4]
    right = dirs[(dirs.index(action) + 1) % 4]

    transitions = []
    for dir_, prob in zip([forward, left, right], [P_FORWARD, P_LEFT, P_RIGHT]):
        dx, dy = action_vectors[dir_]
        ni, nj = i + dx, j + dy
        if not in_bounds(ni, nj):
            ni, nj = i, j  # bounce back
        transitions.append((prob, (ni, nj)))
    return transitions

def policy_evaluation(policy, V):
    while True:
        delta = 0
        new_V = V.copy()
        for i in range(rows):
            for j in range(cols):
                if is_terminal((i, j)):
                    continue
                a = policy[i][j]
                value = 0
                for prob, (ni, nj) in get_transitions((i, j), a):
                    reward = grid[ni][nj] if not is_terminal((ni, nj)) else 0
                    value += prob * (reward + gamma * V[ni][nj])
                new_V[i][j] = value
                delta = max(delta, abs(V[i][j] - value))
        V = new_V
        if delta < threshold:
            break
    return V

def policy_improvement(V, policy):
    stable = True
    for i in range(rows):
        for j in range(cols):
            if is_terminal((i, j)):
                continue
            old_action = policy[i][j]
            action_values = {}
            for a in actions:
                value = 0
                for prob, (ni, nj) in get_transitions((i, j), a):
                    reward = grid[ni][nj] if not is_terminal((ni, nj)) else 0
                    value += prob * (reward + gamma * V[ni][nj])
                action_values[a] = value
            best_action = max(action_values, key=action_values.get)
            policy[i][j] = best_action
            if best_action != old_action:
                stable = False
    return policy, stable

def policy_iteration():
    policy = np.random.choice(actions, size=(rows, cols))
    V = np.zeros((rows, cols))
    for term in [(0, 3), (1, 3)]:
        policy[term] = None
        V[term] = grid[term]  # set terminal state values
    while True:
        V = policy_evaluation(policy, V)
        policy, stable = policy_improvement(V, policy)
        if stable:
            break
    return policy, V

def print_policy(policy):
    for row in policy:
        print(' '.join(['{:>2}'.format(str(a) if a else 'T') for a in row]))

def print_values(V):
    for row in V:
        print(' '.join(['{:>6.2f}'.format(v) for v in row]))

if __name__ == "__main__":
    final_policy, final_values = policy_iteration()
    print("\nOptimal Policy:")
    print_policy(final_policy)
    print("\nValue Function:")
    print_values(final_values)