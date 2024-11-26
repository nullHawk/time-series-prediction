import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import LSTMPredictor
from generate_sinewave import generate, plot


x, y = generate() # y = 100, 10000
# plot(x, y)
train_input = torch.from_numpy(y[3:,:-1]) # 97, 999
train_target = torch.from_numpy(y[3:, 1:]) # 97, 999
test_input = torch.from_numpy(y[:3, :-1]) # 3, 999
test_target = torch.from_numpy(y[:3, 1:]) # 3, 999

model = LSTMPredictor()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(model.parameters(), lr=0.8) # Limited Memory BFGS(Broyden-Fletcher-Goldfarb-Shanno) optimizer, needs closure function

n_steps = 10
for i in range(n_steps):
    print("Step: ", i)

    def closure():
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_target)
        print("Loss: ", loss.item())
        loss.backward()
        return loss
    
    optimizer.step(closure) # closure is a function that reevaluates the model and returns the loss

    with torch.no_grad():
        future = 1000
        pred = model(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print("Test loss: ", loss.item())
        y = pred.detach().numpy()

    plt.figure(figsize=(12, 6))
    plt.title(f"Step {i+1}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    n = train_input.shape[1] # 999

    def draw(y_i, color):
        plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
        plt.plot(np.arange(n, n+future), y_i[n:], color + ":", linewidth=2.0)
    
    draw(y[0], 'r')
    draw(y[1], 'b')
    draw(y[2], 'g')

    plt.savefig("predict%d.pdf"%i)
    plt.close()
        

