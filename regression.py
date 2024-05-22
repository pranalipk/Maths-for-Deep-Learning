import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
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
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.001  # Adjusted learning rate
    num_epochs = 5000  # Increased epochs for better convergence
    input_features = X.shape[1]  # Number of features in input
    output_features = y.shape[1]  # Number of features in output
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    prev_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")
        
        # Early stopping condition
        if abs(prev_loss - loss.item()) < 1e-6:
            break
        
        prev_loss = loss.item()
    
    return model, loss

