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
        return len(state) == self.n and self.is_valid(state)

    def is_valid(self, state):
        """检查 state 中所有皇后是否彼此冲突"""
        n = len(state)  # 现在 state 中皇后的总数

        for i in range(n):  # 遍历所有皇后
            for j in range(i + 1, n):  # 与后面的每个皇后进行比对
                if state[i] == state[j]:  # ✅ 同一列冲突
                    print(f"❌ Invalid state (Same Column Conflict): {state}")
                    return False
                if abs(state[i] - state[j]) == abs(i - j):  # ✅ 对角线冲突
                    print(f"❌ Invalid state (Diagonal Conflict): {state}")
                    return False
        return True  # 没有冲突

    def get_successors(self, state):
        if len(state) >= self.n:
            return []  # put all queens

        successors = []
        row = len(state)  # the row of next queen
        for col in range(self.n):
            new_state = state + [col]
            if self.is_valid(new_state):  # only expend valid state
                successors.append(new_state)
                print(successors)
        return successors

    def get_successors_all(self, state):
        """取得可能的下一個狀態 (擴展節點)"""
        if len(state) >= self.n:
            return []  # put all queens

        successors = []
        row = len(state)  # the row of next queen
        for col in range(self.n):
            new_state = state + [col]
            # **允許產生衝突的狀態**
            successors.append(new_state)
            #print(successors)
        return successors

    def path_cost(self, state):
        return len(state)

# ==============================
# Visualization
# ==============================
def draw_board(state, N, visited, frontier):

    board = [['X' for _ in range(N)] for _ in range(N)]  # build the board

    for f in frontier:
        for row, col in enumerate(f):
            board[row][col] = f"{BLUE}Q{RESET}"  # blue Q for expended node

    for v in visited:
        for row, col in enumerate(v):
            board[row][col] = f"{RED}Q{RESET}"  # red Q for visited

    for row, col in enumerate(state):
        board[row][col] = f"{GREEN}Q{RESET}"  # green Q for finial path



    # 印出棋盤
    print("\n".join([" ".join(row) for row in board]))
    print("=" * (2 * N))
    time.sleep(0.5)  # wait 0.5 s"""
    return board

#===================================================================
def is_in_priority_queue(pq, successor):
    for _, state in pq:
        if state == successor:
            #print("successor already in pq:")
            #print(successor)  # output
            return True  # successor is in pq
    return False  # successor is not in pq

def is_in_priority_queue_a_star(pq, successor):
    for _, _, state in pq:
        if state == successor:
            #print("successor already in pq:")
            #print(successor)  # 找到時輸出
            return True  # successor 存在於 pq
    return False  # successor is not in pq5

def start_game():
    N = int(input("Please Enter the number of Queen you have:"))  # input the number of Queens
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
# BFS (Breadth-First Search)
# ==============================
def bfs(problem):
    
    start_time = time.time()
    frontier = deque([problem.initial_state])   # (state)
    visited = []
    node_explored = 0

    while frontier:

        state = frontier.popleft()
        visited.append(tuple(state))
        #print(f"visited node:{visited}")
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
            print("=" * (2 * problem.n))
            time.sleep(0.5)  # wait 0.5 s

            return visited  # find the path

        #for successor in problem.get_successors(state):
            #if tuple(successor) not in visited:
                #frontier.append(successor)

        for successor in problem.get_successors_all(state):
            if tuple(successor) not in visited:
                frontier.append(successor)
        #print(frontier)
        #print(frontier)
    return None

def dfs(problem):

    start_time = time.time()
    frontier = deque([problem.initial_state])  # (state)
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
            print("=" * (2 * problem.n))
            time.sleep(0.5)  # wait 0.5 s

            return visited

        for successor in reversed(problem.get_successors_all(state)):  # stack
            if tuple(successor) not in visited:
                frontier.append(successor)
        #print(frontier)
    return None

def ucs(problem):

    start_time = time.time()
    pq = []  # priority queue
    heapq.heappush(pq, (0, problem.initial_state))  # (cost, state)
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
            print("=" * (2 * problem.n))
            time.sleep(0.5)  # wait 0.5 s

            return visited

        for successor in problem.get_successors_all(state):
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
                    state_costs[successor_tuple] = new_cost  # update cost

        print(f"priority queue:{pq}")

    return None

def a_star(problem):

    start_time = time.time()
    node_explored = 0
    state_costs = {tuple(problem.initial_state): 0}

    def count_conflicts(state):
        """conflict pairs"""
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    def heuristic(state):
        """heuristic function"""

        return (problem.n - len(state)) +  count_conflicts(state)

    state_h = {tuple(problem.initial_state):heuristic(problem.initial_state)}

    pq = []
    visited = []
    heapq.heappush(pq, (heuristic(problem.initial_state), 0, problem.initial_state))  # (h, cost, state)

    while pq:
        h, cost, state = heapq.heappop(pq)
        visited.append(state)
        print(f"visited node:{visited}")
        node_explored += 1
        pq_states = [s for _, _, s in pq]
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
            print("=" * (2 * problem.n))
            time.sleep(0.5)  # wait 0.5 s

            return visited

        """for successor in problem.get_successors(state):
            new_cost = cost + 1
            heapq.heappush(pq, (new_cost + heuristic(successor), new_cost, successor, path + [state])"""

        for successor in problem.get_successors_all(state):
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
                    print(f"update {successor}  h：{old_h} -> {new_h}")

                    # delete old successor and update
                    pq = [(h, c, s) for h, c, s in pq if s != successor]
                    heapq.heapify(pq)  # re-build min-heap

                    heapq.heappush(pq, (new_h, new_cost, successor))
                    state_h[successor_tuple] = new_h # update cost

        print(f"priority queue:{pq}")
    return None

# ==============================
# 运行程序
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