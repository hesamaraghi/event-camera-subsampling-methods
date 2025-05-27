import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torchvision
from omegaconf.errors import ConfigAttributeError

def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:,None,None,None]

def create_image(representation):
    B, C, H, W = representation.shape
    representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)

    representation = (representation - robust_min_vals)/(robust_max_vals - robust_min_vals)
    representation = torch.clamp(255*representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel

        self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in range(1000):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = events.y.shape[0]
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events.pos[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.pos[:,0], events.pos[:,1], events.pos[:,2], events.x[:,0], events.batch
        
        # normalizing timestamps
        for bi in range(B):
            start = events.ptr[bi]
            end = events.ptr[bi+1]
            if start < end:
                t[start:end] /= t[start:end].max()
            

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x.int() \
                          + W * y.int()\
                          + 0 \
                          + W * H * C * p.int() \
                          + W * H * C * 2 * b.int()

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


def cnn_model(cfg):
    
    input_channels = 2*cfg.model.num_bins
    try:
        cnn_type = cfg.model.cnn_type
        if not cnn_type:
            cnn_type = "resnet34"    
    except ConfigAttributeError:
        cnn_type = "resnet34"


    if cnn_type == "resnet34":
        if not cfg.model.resnet_pretrained:
            weights = None
        else:
            weights = "DEFAULT"
        model = models.get_model(cnn_type, weights=weights)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    
    elif cnn_type == "shufflenet_v2_x0_5":
        if not cfg.model.resnet_pretrained:
            weights = None
        else:
            weights = "DEFAULT" 
        model = models.get_model(cnn_type, weights=weights)   
        model.conv1[0] = nn.Conv2d(input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    
    elif cnn_type == "mobilenet_v3_small":
        if not cfg.model.resnet_pretrained:
            weights = None
        else:
            weights = "DEFAULT" 
        model = models.get_model(cnn_type, weights=weights)
        model.features[0] = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, cfg.dataset.num_classes)
    
    elif cnn_type == "squeezenet1_1":    
        if not cfg.model.resnet_pretrained:
            weights = None
        else:
            weights = "DEFAULT" 
        model = models.get_model(cnn_type, weights=weights)
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2)
        model.classifier[1] = nn.Conv2d(512, cfg.dataset.num_classes, kernel_size=1, stride=1)
    else:
        raise NotImplementedError("CNN type not implemented")
    
    return model
            
class Net(nn.Module):
    def __init__(self, cfg):

        nn.Module.__init__(self)
        self.voxel_dimension = (cfg.model.num_bins,cfg.dataset.image_resolution[0],cfg.dataset.image_resolution[1])
        self.mlp_layers = cfg.model.est_mlp_layers
        self.activation = eval(cfg.model.est_activation)
        self.quantization_layer = QuantizationLayer(self.voxel_dimension, self.mlp_layers, self.activation)
        self.classifier = cnn_model(cfg)

        self.crop_dimension = cfg.model.resnet_crop_dimension

        # replace fc layer and first convolutional layer
        # input_channels = 2*cfg.model.num_bins
        # self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.classifier.fc = nn.Linear(self.classifier.fc.in_features, cfg.dataset.num_classes)

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=tuple(output_resolution))

        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred #, vox


