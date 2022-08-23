def normalize(tensor, eps=1e-9):
    return (tensor - tensor.mean()) / (tensor.std() + eps)