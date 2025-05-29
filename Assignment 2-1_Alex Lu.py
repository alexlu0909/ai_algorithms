"Adversarial Search: Minimax & Alpha_Beta Pruning  Stony Brook University Computer Engineering Alex Lu"
import copy
import math
import time

# Global counters for counting the number of moves attempted by the search algorithms
minimax_moves = 0
alphabeta_moves = 0


# Initialize the board for the checkers game. build a 8x8 initial board.
def initialize_board():

    board = []
    for r in range(8):
        row = []
        for c in range(8):
            if (r + c) % 2 == 0:
                row.append("_")  # "_" represents an unusable square.
            else:
                # Place black pieces ("b") on the top 3 rows and white pieces ("w") on the bottom 3 rows.
                if r < 3:
                    row.append("b")
                elif r > 4:
                    row.append("w")
                else:
                    row.append(".")
        board.append(row)
    return board


# Print the current state of the board.
def print_board(board):
    # Print column numbers
    print("  " + " ".join(str(c) for c in range(8)))
    for r in range(8):
        print(str(r) + " " + " ".join(board[r])) # Print each row with its row number.
    print()


# Create a deep copy of the board.
def copy_board(board):
    return copy.deepcopy(board)


# Switch the player turn.
def to_move(player):
    return "b" if player == "w" else "w"


# Generate moves and validate them (including multi-jump captures).
def get_capture_moves(board, r, c, player, piece, path, captures):

    moves = []
    if piece.isupper():
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] # If the piece is a king(represented by an uppercase letter), it can move in all 4 diagonal directions.
    else:
        directions = [(-1, -1), (-1, 1)] if player == "w" else [(1, -1), (1, 1)]  # For non-king pieces, the movement is limited to forward directions depending on the player.

    # Iterate through each possible direction.
    for dr, dc in directions:
        new_r = r + dr
        new_c = c + dc
        jump_r = r + 2 * dr
        jump_c = c + 2 * dc

        if (0 <= new_r < 8 and 0 <= new_c < 8 and 0 <= jump_r < 8 and 0 <= jump_c < 8):# Check if both the adjacent square and the landing square are within board bounds.
            target = board[new_r][new_c]
            if target != "." and target != "_" and target.lower() == to_move(player) and board[jump_r][jump_c] == ".":  # Check if the adjacent square contains an opponent's piece and the landing square is empty.
                new_board = copy_board(board)
                new_board[r][c] = "."
                new_board[new_r][new_c] = "."
                new_board[jump_r][jump_c] = piece
                new_path = path + [(jump_r, jump_c)]
                new_captures = captures + [(new_r, new_c)]

                keep_jump = get_capture_moves(new_board, jump_r, jump_c, player, piece, new_path, new_captures) # Recursively check for additional jumps (multi-jump capture).
                if keep_jump:
                    moves.extend(keep_jump)
                else:
                    moves.append({"path": new_path, "captures": new_captures})

    return moves


# Generate all possible moves for a given player.
def generate_moves(board, player):

    capturing_moves = []
    non_capture_moves = []
    # Loop through every square on the board.
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece.lower() != player:
                continue
            # First, check for capture moves.
            cap_moves = get_capture_moves(board, r, c, player, piece, [(r, c)], [])
            if cap_moves:
                capturing_moves.extend(cap_moves)
            else:
                # If no capture moves, generate normal moves based on allowed directions.
                if piece.isupper():
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                else:
                    directions = [(-1, -1), (-1, 1)] if player == "w" else [(1, -1), (1, 1)]
                for dr, dc in directions:
                    new_r = r + dr
                    new_c = c + dc
                    if 0 <= new_r < 8 and 0 <= new_c < 8 and board[new_r][new_c] == ".":
                        non_capture_moves.append({"path": [(r, c), (new_r, new_c)], "captures": []})

    # If any capture moves exist, they must be played.
    if capturing_moves:
        return capturing_moves
    else:
        return non_capture_moves


# Apply a move to the board and update the board state.
def apply_move(board, move, player):

    start = move["path"][0]
    end = move["path"][-1]
    piece = board[start[0]][start[1]]
    board[start[0]][start[1]] = "."
    board[end[0]][end[1]] = piece
    # Remove captured pieces.
    for cap in move["captures"]:
        board[cap[0]][cap[1]] = "."
    # If a piece reaches the opponent's end, it becomes a king.
    if player == "w" and end[0] == 0 and piece == "w":
        board[end[0]][end[1]] = "W"
    if player == "b" and end[0] == 7 and piece == "b":
        board[end[0]][end[1]] = "B"


# Check for terminal conditions (game over) when one side has no legal moves or pieces.
def is_terminal(board):

    white_moves = generate_moves(board, "w")
    black_moves = generate_moves(board, "b")
    white_exists = any(board[r][c].lower() == "w" for r in range(8) for c in range(8))
    black_exists = any(board[r][c].lower() == "b" for r in range(8) for c in range(8))
    if not white_exists or not black_exists:
        return True
    if not white_moves or not black_moves:
        return True
    return False


# Evaluation function to assign a score to a board state from the AI's perspective.
def evaluate(board, ai_player):

    score = 0
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == "." or piece == "_":
                continue
            # Normal pieces are valued at 1.0, kings at 2.0.
            value = 1.0 if piece.islower() else 2.0
            if piece.lower() == ai_player:
                score += value
            else:
                score -= value
    return score

# Minimax algorithm: Maximizing part.
def Maximize(board, depth, player, ai_player):
    global minimax_moves
    if depth == 0 or is_terminal(board):
        return evaluate(board, ai_player), None
    moves = generate_moves(board, player)
    if not moves:
        return evaluate(board, ai_player), None

    best_move = None
    max_Utility = -math.inf

    # Iterate over all possible moves.
    for move in moves:
        minimax_moves += 1 # Increase the move counter for minimax search.
        new_board = copy_board(board)
        apply_move(new_board, move, player)
        utility, _ = Minimize(new_board, depth - 1, to_move(player), ai_player)
        if utility > max_Utility:
            max_Utility = utility
            best_move = move
    return max_Utility, best_move


# Minimax algorithm: Minimizing part.
def Minimize(board, depth, player, ai_player):
    global minimax_moves
    if depth == 0 or is_terminal(board):
        return evaluate(board, ai_player), None
    moves = generate_moves(board, player)
    if not moves:
        return evaluate(board, ai_player), None

    best_move = None
    min_Utility = math.inf
    for move in moves:
        minimax_moves += 1  # Increase the move counter for minimax search.
        new_board = copy_board(board)
        apply_move(new_board, move, player)
        utility, _ = Maximize(new_board, depth - 1, to_move(player), ai_player)
        if utility < min_Utility:
            min_Utility = utility
            best_move = move
    return min_Utility, best_move

# Entry function for minimax search.
def Minimax(board, depth, player, ai_player):
    score, best_move = Maximize(board, depth, player, ai_player)
    return score, best_move


# Alpha-Beta pruning: Maximizing part.
def AlphaBeta_Maximize(board, depth, player, ai_player, alpha, beta):
    global alphabeta_moves
    if depth == 0 or is_terminal(board):
        return evaluate(board, ai_player), None
    moves = generate_moves(board, player)
    if not moves:
        return evaluate(board, ai_player), None

    best_move = None
    max_utility = -math.inf
    for move in moves:
        alphabeta_moves += 1 # Increase the move counter for alpha-beta pruning.
        new_board = copy_board(board)
        apply_move(new_board, move, player)
        utility, _ = AlphaBeta_Minimize(new_board, depth - 1, to_move(player), ai_player, alpha, beta)
        if utility > max_utility:
            max_utility = utility
            best_move = move
        alpha = max(alpha, max_utility)
        if beta <= alpha:
            break  # Prune the remaining moves as they cannot improve the outcome.
    return max_utility, best_move


# Alpha-Beta pruning: Minimizing part.
def AlphaBeta_Minimize(board, depth, player, ai_player, alpha, beta):
    global alphabeta_moves
    if depth == 0 or is_terminal(board):
        return evaluate(board, ai_player), None
    moves = generate_moves(board, player)
    if not moves:
        return evaluate(board, ai_player), None

    best_move = None
    min_utility = math.inf
    for move in moves:
        alphabeta_moves += 1 # Increase the move counter for alpha-beta pruning.
        new_board = copy_board(board)
        apply_move(new_board, move, player)
        utility, _ = AlphaBeta_Maximize(new_board, depth - 1, to_move(player), ai_player, alpha, beta)
        if utility < min_utility:
            min_utility = utility
            best_move = move
        beta = min(beta, min_utility)
        if beta <= alpha:
            break  # Prune the remaining moves as they cannot improve the outcome.
    return min_utility, best_move

# Entry function for alpha-beta pruning search.
def AlphaBeta_Minimax(board, depth, player, ai_player):
    return AlphaBeta_Maximize(board, depth, player, ai_player, -math.inf, math.inf)





# Game loop for AI vs Opponent demonstration using the minimax algorithm. Where both sides choose optimal moves.
def game_loop():
    board = initialize_board()
    current_player = "b"  # Assume black moves first.
    ai_player = current_player  # In this example, both sides use minimax.
    max_depth = 6  # Search depth can be adjusted. Initially set to 6 to avoid excessive resource consumption.
    start_time = time.time()
    while not is_terminal(board):
        print_board(board)
        print(f"It is {current_player}'s turn.")
        moves = generate_moves(board, current_player)
        if not moves:
            print(f"{current_player} can not move, GAME OVER.")
            break
        score, best_move = Minimax(board, max_depth, current_player, ai_player)

        if best_move is None:
            print("Unable to find the best move, GAME OVER.")
            break
        print(f"Selected move: from {best_move['path'][0]} to {best_move['path'][-1]}, full path: {best_move['path']}")
        if best_move["captures"]:
            print(f"Capture positions: {best_move['captures']}")

        apply_move(board, best_move, current_player)
        current_player = to_move(current_player)

    print_board(board)
    final_score = evaluate(board, ai_player)
    if final_score > 0:
        end_time = time.time()
        print("AI WINS!")
    elif final_score < 0:
        end_time = time.time()
        print("Opponent WINS!")
    else:
        end_time = time.time()
        print("DRAW！")

    print(f"time-consuming：{end_time - start_time: .2f} sec")
    print(f"Number of moves did <Minimax> attempt: {minimax_moves} times")


# Game loop for AI vs Opponent demonstration using the alpha-beta pruning algorithm. Where both sides choose optimal moves.
def Prune_game_loop():
    board = initialize_board()
    current_player = "b"  # Assume black moves first.
    ai_player = current_player  # In this example, both sides use alpha-beta pruning.
    max_depth = 6  # Search depth can be adjusted. Initially set to 6 to avoid excessive resource consumption.
    start_time = time.time()
    while not is_terminal(board):
        print_board(board)
        print(f"It is {current_player}'s turn.")
        moves = generate_moves(board, current_player)
        if not moves:
            print(f"{current_player} can not move, GAME OVER.")
            break
        score, best_move = AlphaBeta_Minimax(board, max_depth, current_player, ai_player)

        if best_move is None:
            print("Unable to find the best move, GAME OVER.")
            break
        print(f"Selected move: from {best_move['path'][0]} to {best_move['path'][-1]}, full path: {best_move['path']}")
        if best_move["captures"]:
            print(f"Capture positions: {best_move['captures']}")

        apply_move(board, best_move, current_player)
        current_player = to_move(current_player)

    print_board(board)
    final_score = evaluate(board, ai_player)
    if final_score > 0:
        end_time = time.time()
        print("AI WINS!")
    elif final_score < 0:
        end_time = time.time()
        print("Opponent WINS!")
    else:
        end_time = time.time()
        print("DRAW!")

    print(f"time-consuming：{end_time - start_time: .2f} sec")
    print(f"Number of moves did <Alpha-Beta Pruning> attempt: {alphabeta_moves} times")


if __name__ == "__main__":
    print("Minimax:\n")
    game_loop()
    print("----------------------------------------------------------------------")
    print("Alpha_Beta_Pruning:\n")
    Prune_game_loop()