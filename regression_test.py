
import torch
import pytest
from regression import fit_regression_model

def get_train_data(dim=1):
    X_2 = torch.tensor([
        [24., 2.], [24., 4.], [16., 3.], [25., 6.], [16., 1.], 
        [19., 2.], [14., 3.], [22., 2.], [25., 4.], [12., 1.], 
        [24., 7.], [19., 1.], [23., 7.], [19., 5.], [21., 3.], 
        [16., 6.], [24., 5.], [19., 7.], [14., 4.], [20., 3.]
    ])
    y = torch.tensor([
        [1422.40], [1469.50], [1012.70], [1632.20], [952.20], 
        [1117.70], [906.20], [1307.30], [1552.80], [686.70], 
        [1543.40], [1086.50], [1495.20], [1260.70], [1288.10], 
        [1111.50], [1523.10], [1297.40], [946.40], [1197.10]
    ])
    if dim == 1:
        X = X_2[:, :1]
    elif dim == 2:
        X = X_2
    else:
        raise ValueError("dim must be 1 or 2")
    return X, y

def test_fit_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    print(loss)
    assert loss.item() < 4321, "loss too big"

def test_fit_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    assert loss.item() < 400

def test_fit_and_predict_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20.], [15.], [10.]])
    y_pred = model(X_test)
    assert ((y_pred - torch.tensor([[1252.3008], [939.9971], [627.6935]])).abs() < 2).all(), "y_pred is not correct"
    assert y_pred.shape == (3, 1), "y_pred shape is not correct"

def test_fit_and_predict_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20., 2.], [15., 3.], [10., 4.]])
    y_pred = model(X_test)
    assert ((y_pred - torch.tensor([[1191.9037], [943.9369], [695.9700]])).abs() < 2).all(), "y_pred is not correct"
    assert y_pred.shape == (3, 1), "y_pred shape is not correct"

if __name__ == "__main__":
    test_fit_regression_model_1d()
    test_fit_regression_model_2d()
    test_fit_and_predict_regression_model_1d()
    test_fit_and_predict_regression_model_2d()
