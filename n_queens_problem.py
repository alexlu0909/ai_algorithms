from collections import deque
import heapq
import time


RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

class NQueensProblem:
    def __init__(self, n, f):
        self.n = n  # Board size (N)
        self.f = f
        self.initial_state = [f]  # initial state (board with one position that user selected)

    def is_goal(self, state):
        """檢查是否達到目標狀態 (所有 N 個皇后都放置且不衝突)"""
        return len(state) == self.n and self.is_valid(state)

    def is_valid(self, state):
        """檢查當前狀態是否為合法 (不衝突)"""
        row = len(state) - 1
        for i in range(row):
            diff = abs(state[i] - state[row])
            if diff == 0 or diff == row - i:
                return False
        return True

    def get_successors(self, state):
        """取得可能的下一個狀態 (擴展節點)"""
        if len(state) >= self.n:
            return []  # 如果已放滿 N 個皇后，則沒有後繼狀態

        successors = []
        row = len(state)  # 下一個要放置皇后的行數
        for col in range(self.n):
            new_state = state + [col]
            if self.is_valid(new_state):  # 只加入合法的狀態
                successors.append(new_state)
                print(successors)
        return successors

    def a_star_get_successors(self, state):
        """取得可能的下一個狀態 (擴展節點)"""
        if len(state) >= self.n:
            return []  # 如果已放滿 N 個皇后，則沒有後繼狀態

        successors = []
        row = len(state)  # 下一個要放置皇后的行數
        for col in range(self.n):
            new_state = state + [col]
            # **允許產生衝突的狀態**
            successors.append(new_state)
            print(successors)
        return successors

    def path_cost(self, state):
        """計算當前狀態的路徑成本 (每次放置皇后 +1)"""
        return len(state)

# ==============================
# Visualization
# ==============================
def draw_board(state, N, visited, frontier):
    """以文本方式顯示棋盤，標示皇后、探索過的節點和空白位置"""
    board = [['X' for _ in range(N)] for _ in range(N)]  # 預設為 'X'

    # 標記還在 frontier 內的節點（黃色 Q）
    for f in frontier:
        for row, col in enumerate(f):
            board[row][col] = f"{BLUE}Q{RESET}"  # 用黃色表示還在 frontier 的節點

    # 標記已訪問的節點（紅色 Q）
    for v in visited:
        for row, col in enumerate(v):
            board[row][col] = f"{RED}Q{RESET}"  # 訪問過的節點標紅色

    # 標記皇后位置
    for row, col in enumerate(state):
        board[row][col] = f"{GREEN}Q{RESET}"  # Q 代表皇后



    # 印出棋盤
    print("\n".join([" ".join(row) for row in board]))#############
    print("=" * (2 * N))  # 分隔線#################################
    time.sleep(0.5)  # 等待一秒，以便顯示動畫效果#####################"""
    return board

#=================================================================== 遍歷pq
def is_in_priority_queue(pq, successor):
    for _, state in pq:
        if state == successor:
            print("successor already in pq:")
            print(successor)  # 找到時輸出
            return True  # successor 存在於 pq
    return False  # successor 不在 pq

def is_in_priority_queue_a_star(pq, successor):
    for _, _, state in pq:
        if state == successor:
            print("successor already in pq:")
            print(successor)  # 找到時輸出
            return True  # successor 存在於 pq
    return False  # successor 不在 pq

# ==============================
# BFS (Breadth-First Search)
# ==============================
def bfs(problem):
    """使用 BFS 來解 N-Queens"""
    start_time = time.time()
    frontier = deque([problem.initial_state])   # (狀態, 路徑)
    visited = []
    node_explored = 0

    while frontier:

        state = frontier.popleft()
        visited.append(tuple(state))
        print(f"visited node:{visited}")
        node_explored += 1
        board = draw_board(state, problem.n, visited, frontier)
        print(f"node_explored num:{node_explored}")
        if problem.is_goal(state):
            run_time = time.time() - start_time
            print(f"Searching time: {run_time:.6f}s")
            print(f"How many nodes have tried: {node_explored}")
            print("=" * (2 * problem.n))
            print("final path:")

            for row in range(problem.n):
                for col in range(problem.n):
                    if board[row][col] == f"{RED}Q{RESET}" or board[row][col] == f"{BLUE}Q{RESET}":
                        board[row][col] = 'X'

            print("\n".join([" ".join(row) for row in board]))
            print("=" * (2 * problem.n))  # 分隔線
            time.sleep(0.5)  # 等待一秒，以便顯示動畫效果

            return visited  # 找到解

        for successor in problem.get_successors(state):
            if tuple(successor) not in visited:
                frontier.append(successor)
        print(frontier)


    return None  # 找不到解"""


# ==============================
# DFS (Depth-First Search)
# ==============================
def dfs(problem):
    """使用 DFS 來解 N-Queens"""
    start_time = time.time()
    frontier = deque([problem.initial_state])  # (狀態, 路徑)
    visited = []
    node_explored = 0

    while frontier:
        state = frontier.pop()
        visited.append(state)
        print(f"visited node:{visited}")
        node_explored += 1
        board = draw_board(state, problem.n, visited, frontier)
        print(f"node_explored num:{node_explored}")

        if problem.is_goal(state):
            run_time = time.time() - start_time
            print(f"Searching time: {run_time:.6f}s")
            print(f"How many nodes have tried: {node_explored}")
            print("=" * (2 * problem.n))
            print("final path:")

            for row in range(problem.n):
                for col in range(problem.n):
                    if board[row][col] == f"{RED}Q{RESET}" or board[row][col] == f"{BLUE}Q{RESET}":
                        board[row][col] = 'X'

            print("\n".join([" ".join(row) for row in board]))
            print("=" * (2 * problem.n))  # 分隔線
            time.sleep(0.5)  # 等待一秒，以便顯示動畫效果

            return visited

        for successor in reversed(problem.get_successors(state)):  # 反向擴展，符合 DFS
            if tuple(successor) not in visited:
                frontier.append(successor)
        print(frontier)
    return None


# ==============================
# UCS (Uniform-Cost Search)
# ==============================
def ucs(problem):
    """使用 UCS 來解 N-Queens"""
    start_time = time.time()
    pq = []  # 優先佇列
    heapq.heappush(pq, (0, problem.initial_state))  # (成本, 狀態, 路徑)
    visited = []
    node_explored = 0
    state_costs = {tuple(problem.initial_state): 0}


    while pq:
        cost, state = heapq.heappop(pq)
        visited.append(state)
        print(f"visited node:{visited}")
        node_explored += 1
        pq_states = [s for _, s in pq]
        board = draw_board(state, problem.n, visited, pq_states)
        print(f"node_explored num:{node_explored}")

        if problem.is_goal(state):
            run_time = time.time() - start_time
            print(f"Searching time: {run_time:.6f}s")
            print(f"How many nodes have tried: {node_explored}")
            print("=" * (2 * problem.n))
            print("final path:")

            for row in range(problem.n):
                for col in range(problem.n):
                    if board[row][col] == f"{RED}Q{RESET}" or board[row][col] == f"{BLUE}Q{RESET}":
                        board[row][col] = 'X'

            print("\n".join([" ".join(row) for row in board]))
            print("=" * (2 * problem.n))  # 分隔線
            time.sleep(0.5)  # 等待一秒，以便顯示動畫效果

            return visited

        for successor in problem.get_successors(state):
            new_cost = cost + 1
            successor_tuple = tuple(successor)
            if successor_tuple not in visited and not is_in_priority_queue(pq, successor):
                heapq.heappush(pq, (new_cost, successor))
                state_costs[successor_tuple] = new_cost
            elif is_in_priority_queue(pq, successor):
                old_cost = state_costs.get(successor_tuple, float('inf'))
                if new_cost < old_cost:
                    print(f"更新 {successor} 的 cost：{old_cost} -> {new_cost}")

                    # 刪除舊的 successor 並更新
                    pq = [(c, s) for c, s in pq if s != successor]
                    heapq.heapify(pq)
                    heapq.heappush(pq, (new_cost, successor))
                    state_costs[successor_tuple] = new_cost  # 更新成本紀錄

        print(f"priority queue:{pq}")

    return None


# ==============================
# A* (A-Star Search)
# ==============================
def a_star(problem):
    """使用 A* 來解 N-Queens"""
    start_time = time.time()
    node_explored = 0
    state_costs = {tuple(problem.initial_state): 0}

    def count_conflicts(state):
        """計算皇后間的衝突數"""
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    def heuristic(state):
        """啟發式函數 (計算當前皇后數量，目標是 N 個)"""

        return (problem.n - len(state)) +  count_conflicts(state)

    state_h = {tuple(problem.initial_state):heuristic(problem.initial_state)}

    pq = []
    visited = []
    heapq.heappush(pq, (heuristic(problem.initial_state), 0, problem.initial_state))  # (h值, 成本, 狀態, 路徑)

    while pq:
        h, cost, state = heapq.heappop(pq)
        visited.append(state)
        node_explored += 1
        pq_states = [s for _, _, s in pq]
        board = draw_board(state, problem.n, visited, pq_states)

        if problem.is_goal(state):
            run_time = time.time() - start_time
            print(f"Searching time: {run_time:.6f}s")
            print(f"How many nodes have tried: {node_explored}")
            print("=" * (2 * problem.n))
            print("final path:")

            for row in range(problem.n):
                for col in range(problem.n):
                    if board[row][col] == f"{RED}Q{RESET}" or board[row][col] == f"{BLUE}Q{RESET}":
                        board[row][col] = 'X'

            print("\n".join([" ".join(row) for row in board]))
            print("=" * (2 * problem.n))  # 分隔線
            time.sleep(0.5)  # 等待一秒，以便顯示動畫效果

            return visited

        """for successor in problem.get_successors(state):
            new_cost = cost + 1
            heapq.heappush(pq, (new_cost + heuristic(successor), new_cost, successor, path + [state])"""

        for successor in problem.a_star_get_successors(state):
            new_cost = cost + 1
            successor_tuple = tuple(successor)
            if successor_tuple not in visited and not is_in_priority_queue_a_star(pq, successor):
                heapq.heappush(pq, (new_cost + heuristic(successor), new_cost, successor))
                state_costs[successor_tuple] = new_cost
                state_h[successor_tuple] = new_cost + heuristic(successor)
            elif is_in_priority_queue_a_star(pq, successor):
                old_h = state_h.get(successor_tuple, float('inf'))
                new_h = new_cost + heuristic(successor)
                if new_h < old_h:
                    print(f"更新 {successor} 的 h：{old_h} -> {new_h}")

                    # 刪除舊的 successor 並更新
                    pq = [(h, c, s) for h, c, s in pq if s != successor]
                    heapq.heapify(pq)  # 重新建立 min-heap

                    heapq.heappush(pq, (new_h, new_cost, successor))
                    state_h[successor_tuple] = new_h # 更新成本紀錄

        print(f"priority queue:{pq}")



    return None


def start_game():
    N = int(input("Please Enter the number of Queen you have:"))  # 設定 N-Queens 的大小
    input_position = False
    while not input_position:
        first_queen = int(input(f"Please Enter the first Queen's position(1~{N}): "))
        if first_queen > N:
            print("Invalid input! Position must be between 1 and", N)
        else:
            input_position = True
    print("Start putting Queen from the top of board")
    problem = NQueensProblem(N, first_queen-1)

    return problem


# ==============================
# 測試 N-Queens 解法
# ==============================
if __name__ == "__main__":
    problem = start_game()

    print("=== BFS ===")
    solution_bfs = bfs(problem)
    print(solution_bfs[-1] if solution_bfs else "No solution found")

    print("\n=== DFS ===")
    solution_dfs = dfs(problem)
    print(solution_dfs[-1] if solution_dfs else "No solution found")

    print("\n=== UCS ===")
    solution_ucs = ucs(problem)
    print(solution_ucs[-1] if solution_ucs else "No solution found")

    print("\n=== A* ===")
    solution_a_star = a_star(problem)
    print(solution_a_star[-1] if solution_a_star else "No solution found")
