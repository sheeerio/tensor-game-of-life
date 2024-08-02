import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def kernel(T):
    kernel = torch.ones((1, 1, T, T), dtype=torch.float32)
    kernel[:,:,T//2,T//2] = 0
    return kernel

def update(board, kernel, birth, survive):
    n = F.conv2d(board, kernel, padding='same')
    _birth = torch.stack([(n == b) for b in birth]).any(0)
    _survive = torch.stack([(n == s) for s in survive]).any(0)
    board_ = torch.where(board == 1, _survive, _birth)
    return board_.float()

def sim(board, *args, **kwargs):
    print(f"kernel: \n{args[0].numpy()} \nbirth: {args[1]} \nsurvive: {args[2]}")
    _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(board.squeeze().cpu().numpy(), cmap='binary')
    for _ in range(kwargs['S']):
        board = update(board, *args)
        im.set_data(board.squeeze().cpu().numpy())
        plt.pause(0.1)
    plt.show()

T = 3
birth = [T]
survive = [T-1, T]
N = 100

if __name__ == '__main__':
    kernel = kernel(T)
    board = torch.randint(0, 2, (1, 1, N, N), dtype=torch.float32)
    sim(board, kernel, birth, survive, S=500)