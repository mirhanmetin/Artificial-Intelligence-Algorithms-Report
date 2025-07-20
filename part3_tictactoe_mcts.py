import copy
import math
import random

EMPTY = '.'
PLAYER_X = 'X'
PLAYER_O = 'O'

class GameState:
    def __init__(self, board=None, player=PLAYER_X):
        self.board = board if board else [[EMPTY]*3 for _ in range(3)]
        self.player = player

    def clone(self):
        return GameState(copy.deepcopy(self.board), self.player)

    def get_legal_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == EMPTY]

    def make_move(self, move):
        i, j = move
        self.board[i][j] = self.player
        self.player = PLAYER_O if self.player == PLAYER_X else PLAYER_X

    def get_winner(self):
        lines = self.board + list(zip(*self.board)) + [
            [self.board[i][i] for i in range(3)],
            [self.board[i][2 - i] for i in range(3)]
        ]
        for line in lines:
            if line[0] != EMPTY and all(cell == line[0] for cell in line):
                return line[0]
        if all(cell != EMPTY for row in self.board for cell in row):
            return 'Draw'
        return None

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_legal_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.41):
        return max(self.children, key=lambda child:
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
        )

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.clone()
        next_state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        if result == self.state.player:
            self.wins += 0
        elif result == 'Draw':
            self.wins += 0.5
        else:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

def smart_simulate_game(state):
    while not state.get_winner():
        legal = state.get_legal_moves()
        for move in legal:
            test = state.clone()
            test.make_move(move)
            if test.get_winner() == state.player:
                state.make_move(move)
                break
        else:
            for move in legal:
                test = state.clone()
                test.make_move(move)
                if test.get_winner() != None and test.get_winner() != state.player:
                    state.make_move(move)
                    break
            else:
                state.make_move(random.choice(legal))
    return state.get_winner()


def mcts_enhanced(state, iter_limit=2000, rollout_count=10):
    root = MCTSNode(state)
    for _ in range(iter_limit):
        node = root
        temp_state = state.clone()

        # Select
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            temp_state.make_move(node.move)

        # Expand
        if node.untried_moves:
            node = node.expand()
            temp_state.make_move(node.move)

        # Simulate multiple rollouts
        results = {'X': 0, 'O': 0, 'Draw': 0}
        for _ in range(rollout_count):
            result = smart_simulate_game(temp_state)
            results[result] += 1

        # Aggregate result: majority vote
        final_result = max(results, key=results.get)
        node.backpropagate(final_result)

    # Selective search: best root move
    return root.best_child(c_param=0).move

def play_game():
    state = GameState()
    print("Welcome to MCTS Tic-Tac-Toe. You are X. AI is O.")
    state.print_board()

    while state.get_winner() is None:
        if state.player == PLAYER_X:
            move = input("Your move (row,col): ")
            try:
                i, j = map(int, move.strip().split(','))
                if state.board[i][j] != EMPTY:
                    print("Invalid move. Try again.")
                    continue
            except:
                print("Format error. Use: row,col (e.g. 1,2)")
                continue
        else:
            print("AI is thinking...")
            i, j = mcts_enhanced(state, iter_limit=3000, rollout_count=10)
            print(f"AI plays: {i},{j}")

        state.make_move((i, j))
        state.print_board()

    result = state.get_winner()
    if result == 'Draw':
        print("It's a draw!")
    else:
        print(f"{result} wins!")

if __name__ == '__main__':
    play_game()
