import torch

def create_tensor_of_val(dimensions, val):
    res = torch.ones(dimensions) * val
    return res

def calculate_elementwise_product(A, B):
    res = A * B
    return res

def calculate_matrix_product(X, W):
    res = torch.matmul(X, W.T)
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    res = torch.matmul(X, W.T) + b
    return res

def calculate_activation(sum_total):
    res = torch.heaviside(sum_total, torch.tensor(0.0))
    return res

def calculate_output(X, W, b):
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    res = calculate_activation(sum_total)
    return res
