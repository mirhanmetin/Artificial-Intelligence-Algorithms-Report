import numpy as np
import random

# GridWorld setup
rows, cols = 4, 4
terminal_state = (0, 3)

# Randomly generate 2 blocking cells (excluding terminal)
all_states = [(i, j) for i in range(rows) for j in range(cols) if (i, j) != terminal_state]
blocked_states = random.sample(all_states, 2)

actions = ['U', 'D', 'L', 'R']
action_idx = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
action_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Q-learning parameters
epsilon = 0.1  # exploration rate
alpha = 0.5    # learning rate
gamma = 0.9    # discount factor
episodes = 3000

# Initialize Q-table
Q = np.zeros((rows, cols, len(actions)))

# Reward function
def get_reward(state):
    if state == terminal_state:
        return 1
    elif state in blocked_states:
        return -1  # penalty if somehow entered (shouldn’t happen)
    else:
        return -0.04  # small negative reward to encourage shortest path

def in_bounds(i, j):
    return 0 <= i < rows and 0 <= j < cols and (i, j) not in blocked_states

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))
    else:
        i, j = state
        return np.argmax(Q[i, j])

def get_next_state(state, action):
    i, j = state
    di, dj = action_vectors[action]
    ni, nj = i + di, j + dj
    if in_bounds(ni, nj):
        return (ni, nj)
    return state

# Training loop
for episode in range(episodes):
    state = (3, 0)  # start in bottom-left corner
    while state != terminal_state:
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        i, j = state
        ni, nj = next_state
        best_next = np.max(Q[ni, nj])
        Q[i, j, action] += alpha * (reward + gamma * best_next - Q[i, j, action])
        state = next_state

# Derive policy
policy = [['' for _ in range(cols)] for _ in range(rows)]
for i in range(rows):
    for j in range(cols):
        if (i, j) in blocked_states:
            policy[i][j] = 'X'
        elif (i, j) == terminal_state:
            policy[i][j] = 'T'
        else:
            best_action = np.argmax(Q[i, j])
            policy[i][j] = actions[best_action]

# Display Q-values and Policy
print("\nBlocking States:", blocked_states)
print("\nLearned Q-Table:")
for i in range(rows):
    for j in range(cols):
        print(f"({i},{j}):", np.round(Q[i, j], 2))
    print()

print("\nDerived Policy:")
for row in policy:
    print(' '.join(row))
