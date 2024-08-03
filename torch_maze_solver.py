import torch
import torch.nn.functional as F
import numpy as np
import time

def create_directional_kernels(K):
    middle = K // 2
    kernels = []
    for i in range(K):
        for j in range(K):
            if (i == middle and j == middle) or (abs(i - middle) + abs(j - middle) != 1):
                continue
            kernel = torch.zeros((K, K))
            kernel[middle, middle] = 1
            kernel[i, j] = 1
            kernels.append(kernel)
    return torch.stack(kernels).unsqueeze(1)

def solve_maze(maze, K):
    maze = torch.sparse_coo_tensor(
        indices=torch.nonzero(maze).t(),
        values=torch.ones(maze.nonzero().shape[0]),
        size=maze.shape
    )
    
    kernels = create_directional_kernels(K)
    state = torch.zeros_like(maze.to_dense())
    state[0, 0] = 1
    state = state.unsqueeze(0).unsqueeze(0)

    while True:
        propagation = F.conv2d(state, kernels, padding=K//2)
        state_ = torch.max(propagation, dim=1, keepdim=True)[0]
        state_ = state_ * maze.to_dense().unsqueeze(0).unsqueeze(0)
        
        if state_[0, 0, -1, -1] > 0:
            return True
        if torch.all(state_ == state):
            return False
        state = state_

if __name__ == '__main__':
    N = 10
    maze = torch.tensor(np.random.choice([0, 1], size=(N, N), p=[0.3, 0.7]))
    maze[0, 0], maze[N-1, N-1] = 1, 1
    print(f"Maze: \n{maze.numpy()}")
    for K in [3, 5, 7]:
        print(f"\nTesting with K = {K}")
        start = time.time()
        result = solve_maze(maze, K)
        print(f"Maze is {'solvable' if result else 'not solvable'}")
        print(f"time: {(time.time() - start) * 1e3:.2f} ms")