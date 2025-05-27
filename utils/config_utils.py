from glob import glob
import os
import os.path as osp
from pytorch_lightning.loggers import WandbLogger
import wandb
from omegaconf import OmegaConf
import yaml
import numpy as np
import json

def show_cfg(cfg):
    """
    Print the configuration in a readable format.
    """

    print(yaml.dump(OmegaConf.to_container(cfg), default_flow_style=False))
                #  sort_keys=True, indent=4))

def recursive_dict_compare(all_cfg, other_cfg):
    """
    Recursively compare two dictionaries and return their differences.
    """

    
    # Initialize the result dictionary
    diff = {}

    # Check for keys in dict1 that are not in dict2
    for key in other_cfg:
        if key not in all_cfg:
            diff[key] = other_cfg[key]
        else:
            # If the values are dictionaries, recursively compare them
            if isinstance(all_cfg[key], dict) and isinstance(other_cfg[key], dict):
                nested_diff = recursive_dict_compare(all_cfg[key], other_cfg[key])
                if nested_diff:
                    diff[key] = nested_diff
            # Otherwise, compare the values directly
            elif all_cfg[key] != other_cfg[key]:
                if not(key == "num_classes" and other_cfg[key] is None and all_cfg[key] is not None):
                    diff[key] = other_cfg[key]
                    

    return diff

def get_checkpoint_file(entity,project,run_id,remote_root = None):
    """
    Load the checkpoint from the specified run_id
    """
    if remote_root is  None:
        checkpoint_file = glob(osp.join('log_folder', project, run_id, "checkpoints","*"))
    elif osp.exists(remote_root):
        checkpoint_file = glob(osp.join(remote_root, 'log_folder', project, run_id, "checkpoints","*"))
    else:
        raise Warning("remote_root does not exist!")
    if checkpoint_file:
        # assert len(checkpoint_file) == 1
        if len(checkpoint_file) > 1:
            print("Multiple checkpoints found, loading the latest one!")
        checkpoint_file = sorted(checkpoint_file, key = lambda x: x.split(os.sep)[-1].split(".")[0], reverse = True)[0]
        print("loading checkpoint from", checkpoint_file)
    else:    
        checkpoint_file = glob(osp.join(project, run_id, "checkpoints","*"))
        if checkpoint_file:
            assert len(checkpoint_file) == 1
            checkpoint_file = checkpoint_file[0]
            print("loading checkpoint from", checkpoint_file)
        else:
            checkpoint_file = glob(osp.join(run_id, "checkpoints","*"))
            if checkpoint_file:
                assert len(checkpoint_file) == 1
                checkpoint_file = checkpoint_file[0]
                print("loading checkpoint from", checkpoint_file)
            else:
                checkpoint_file = glob(osp.join('DGCNN',run_id, "checkpoints","*"))
                if checkpoint_file:
                    assert len(checkpoint_file) == 1
                    checkpoint_file = checkpoint_file[0]
                    print("loading checkpoint from", checkpoint_file)
                else:
                    print("loading checkpoint from wandb server:")
                    checkpoint_folder = WandbLogger.download_artifact(artifact=osp.join(entity, project, f"model-{run_id}:latest"))
                    checkpoint_file = glob(osp.join(checkpoint_folder,"*.ckpt"))
                    if checkpoint_file:
                        assert len(checkpoint_file) == 1
                        checkpoint_file = checkpoint_file[0]
                        print("loading checkpoint from", checkpoint_file) 
                    else:
                        raise ValueError("Attemps failed in finding checkpoint file!")
    
    return checkpoint_file

def get_config_file(entity,project,run_id, verbose = True, cfg_bare = None):
    """
    Load the config from the specified run_id
    """
    
    api = wandb.Api()
    if cfg_bare is None:
        cfg_bare = OmegaConf.load("config_bare.yaml")
    config = api.run(osp.join(entity, project, run_id)).config
    cfg = OmegaConf.create(config) 

    if "cfg_path" in cfg.keys():
        print(cfg.cfg_path)
        try:
            cfg_file = OmegaConf.merge(cfg_bare,OmegaConf.load(cfg.cfg_path))
        except FileNotFoundError:
            cfg_file = cfg
    else:
        cfg_file = cfg
    cfg = OmegaConf.merge(cfg_file, cfg)
    if verbose:
        print(50*"=")
        print("cfg_file")
        print(50*"-")
        print(yaml.dump(recursive_dict_compare(OmegaConf.to_object(cfg),OmegaConf.to_object(cfg_file)), default_flow_style=False))
        print(50*"=")
        print("cfg")
        print(50*"-")
        print(yaml.dump(recursive_dict_compare(OmegaConf.to_object(cfg_file),OmegaConf.to_object(cfg)), default_flow_style=False))
    
    return cfg, cfg_file

def relative_symlink(src, dst):
    """
    Create a relative symlink from src to dst
    """
    src = osp.relpath(src, osp.dirname(dst))
    try:
        os.symlink(src, dst)
    except FileExistsError:
        os.unlink(dst)
        os.symlink(src, dst)
        
def train_validation_test_split_with_number(indices, class_dict, train_percentage = 0.75, validation_percentage = 0.10, num_samples_per_class = None):

    """
    Splits the given indices into train, validation, and test sets based on the specified percentages.

    Args:
        indices (numpy.ndarray): The array of indices to be split.
        class_dict (dict): A dictionary mapping class labels to their corresponding indices.
        train_percentage (float, optional): The percentage of data to be used for training. Defaults to 0.75.
        validation_percentage (float, optional): The percentage of data to be used for validation. Defaults to 0.10.
        num_samples_per_class (int, optional): The maximum number of samples per class to consider. Defaults to None.

    Returns:
        tuple: A tuple containing three numpy arrays: train_indices, validation_indices, and test_indices.
    """    

    data_indices = np.arange(len(indices))
    np.random.shuffle(data_indices)
    indices = indices[data_indices]
    # data_indices = np.arange(len(indices))

    train_indices = np.array([],dtype=np.int64)
    validation_indices = np.array([],dtype=np.int64)
    test_indices = np.array([],dtype=np.int64)

    for class_index in np.arange(len(class_dict)):

        data_class_indices = data_indices[indices == class_index]
        total_num = len(data_class_indices)
        if num_samples_per_class is not None:
            num_samples_per_class = np.minimum(num_samples_per_class, total_num)
            data_class_indices = data_class_indices[:num_samples_per_class]

        tresh_num = int(len(data_class_indices) * train_percentage)
        tresh_num_2 = int(len(data_class_indices) * (train_percentage + validation_percentage))
        train_indices = np.append(train_indices, data_class_indices[:tresh_num])
        validation_indices = np.append(validation_indices, data_class_indices[tresh_num:tresh_num_2])
        test_indices = np.append(test_indices, data_class_indices[tresh_num_2:])

    return train_indices, validation_indices, test_indices