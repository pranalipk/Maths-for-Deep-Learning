import torch
from torch import nn
import torch.nn.functional as F

def create_linear_regression_model(input_size, output_size):
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    learning_rate = 0.001  # Start with a smaller learning rate
    num_epochs = 5000  # Increase the number of epochs
    input_features = X.shape[1]
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    prev_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        if epoch % 1000 == 0:  # Print loss every 1000 epochs
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        # Check for convergence (simple early stopping)
        if abs(prev_loss - loss.item()) < 1e-6:
            print(f'Converged at epoch {epoch}')
            break
        prev_loss = loss.item()
    
    return model, loss

