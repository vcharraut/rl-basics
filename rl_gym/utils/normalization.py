def normalize(tensor, eps=1e-9):
    """_summary_

    Args:
        tensor (_type_): _description_
        eps (_type_, optional): _description_. Defaults to 1e-9.

    Returns:
        _type_: _description_
    """

    return (tensor - tensor.mean()) / (tensor.std() + eps)
