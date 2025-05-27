from matplotlib import animation
import numpy as np 
import matplotlib.pyplot as plt
from utils.data_utils import is_numpy_event_data, is_pyg_event_data, pyg2numpy_event_convertor
import torch

def animate_events(events, transform, fig_size=None):
    
    if is_pyg_event_data(events):
        events = pyg2numpy_event_convertor(events)
    else:
        assert is_numpy_event_data(events), "The input data must be a structured numpy array or a PyG dataset."
    
    frames = transform(events)
    if fig_size is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig, ax = plt.subplots(1,1, figsize=fig_size)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 0, ...] = frames[:, 0, ...] > 0
        rgb[:, 2, ...] = frames[:, 1, ...] > 0
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)
    im = ax.imshow(frames[0])
    # plt.axis("off")
    # plt.axis("tight")
    # ax.invert_yaxis()
    def animate(frame):
        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.close(fig)

    return anim
        
def put_event_side_by_side(datas, image_size, axis = 0):
    assert isinstance(datas, list), "The input data must be a list of PyG datasets."
    assert len(datas) > 1, "The input data must contain at least two PyG datasets."
    num_data = len(datas)
    data_combined = datas[0].clone()
    for i in range(1,num_data):  
        data_to_be_shifted = datas[i].clone()  
        data_to_be_shifted.pos[:,axis] += i * image_size[axis]
        data_combined.pos = torch.cat([data_combined.pos, data_to_be_shifted.pos], dim=0)
        data_combined.x = torch.cat([data_combined.x, data_to_be_shifted.x], dim=0)
    sorted_indices = torch.argsort(data_combined.pos[..., -1])
    data_combined.pos = data_combined.pos[sorted_indices]
    data_combined.x = data_combined.x[sorted_indices]
    return data_combined