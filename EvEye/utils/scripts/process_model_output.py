import torch


def process_model_output(output, use_softmax=True):
    """
    Process the model output to get the predicted class

    Args:
        output (torch.Tensor): The model output tensor
        use_softmax (bool): Whether to apply softmax to the output tensor
    Returns:
        torch.Tensor: The predicted class
    """
    if use_softmax:
        output = torch.softmax(output, dim=1)
    return output.argmax(dim=1).squeeze()
