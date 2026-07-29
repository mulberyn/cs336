import torch
import numpy.typing as npt


def data_loading(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
):
    inputs = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    targets = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        start_idx = torch.randint(0, len(dataset) - context_length, (1, )).item()
        inputs[i] = torch.tensor(
            dataset[start_idx: start_idx + context_length], 
            dtype=torch.long
        )
        targets[i] = torch.tensor(
            dataset[start_idx + 1: start_idx + context_length + 1], 
            dtype=torch.long
        )
    
    return inputs, targets