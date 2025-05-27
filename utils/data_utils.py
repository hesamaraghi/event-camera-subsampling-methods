import numpy as np
import torch
from torch_geometric.data import  Data
from tonic.io import make_structured_array

def pyg2numpy_event_convertor(data):
    """
    Convert event data from a PyG dataset to a structured numpy arrays.
    Args:
        data (torch_geometric.data.Data): The PyG dataset to be converted.
    Returns:
        structured numpy array: The structured numpy array containing the event data.
    """
    
    # dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    events_struct = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)]
    )
    
    return make_structured_array(
        data.pos[:,0],
        data.pos[:,1],
        data.pos[:,-1],
        (data.x[:,0].squeeze() + 1 ) / 2,
        dtype=events_struct,
    )
    


def numpy2pyg_event_convertor(events):
    """
    Convert event data from a structured numpy array to a PyG dataset.
    Args:
        events (structured numpy array): The structured numpy array containing the event data.
    Returns:
        torch_geometric.data.Data: The PyG dataset to be converted.
    """
    
    pos = np.concatenate(
        [
            events["x"].reshape(-1,1).astype(np.float32),
            events["y"].reshape(-1,1).astype(np.float32),
            events["t"].reshape(-1,1).astype(np.float32),
        ], 
        axis=1)
    pos = torch.from_numpy(pos)
    data_p = events["p"].reshape(-1,1).astype(np.float32) * 2 - 1.0 # Convert from [0,1] to [-1,1]
    data_p = torch.from_numpy(data_p)
    data = Data(x=data_p,pos=pos)
    return data

def is_numpy_event_data(data):
    """
    Check if the input data is a structured numpy array.
    Args:
        data (Any): The input data to be checked.
    Returns:
        bool: True if the input data is a structured numpy array, False otherwise.
    """
    
    if isinstance(data, np.ndarray) and data.dtype.names is not None:
        assert "x" in data.dtype.names, "The structured numpy array must have a field named 'x'."
        assert "y" in data.dtype.names, "The structured numpy array must have a field named 'y'."
        assert "t" in data.dtype.names, "The structured numpy array must have a field named 't'."
        assert "p" in data.dtype.names, "The structured numpy array must have a field named 'p'."
        return True
    return False

def is_pyg_event_data(data):
    """
    Check if the input data is a PyG dataset.
    Args:
        data (Any): The input data to be checked.
    Returns:
        bool: True if the input data is a PyG dataset, False otherwise.
    """
    
    if isinstance(data, Data):
        assert "pos" in data, "The PyG dataset must have a field named 'pos'."
        assert "x" in data, "The PyG dataset must have a field named 'x'."
        return True
    return False