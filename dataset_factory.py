from dataloaders.nmnist_loader import nmnist_train_loader, nmnist_test_loader
from dataloaders.ncaltech101_loader import ncaltech101_train_loader, ncaltech101_validation_loader, ncaltech101_test_loader
from dataloaders.nasl_loader import nasl_train_loader, nasl_validation_loader, nasl_test_loader
from dataloaders.ncars_loader import ncars_train_loader, ncars_validation_loader, ncars_test_loader

def factory(cfg):
    """
    Returns:
        train_loader: Training dataset loader.
        val_loader: Validation dataset loader. Validation is performed after
            each training epoch. If None, no validation is performed.
        test_loader: Test dataset loader. Testing is performed after fitting is
            done. If None, no testing is performed.
    """
    if cfg.dataset.name == 'NMNIST':
        cfg.dataset.num_classes = nmnist_train_loader(cfg).dataset.num_classes
        return nmnist_train_loader(cfg), nmnist_test_loader(cfg), None
    if cfg.dataset.name == 'NCALTECH101':
        cfg.dataset.num_classes = ncaltech101_train_loader(cfg).dataset.num_classes
        return ncaltech101_train_loader(cfg), ncaltech101_validation_loader(cfg), ncaltech101_test_loader(cfg)
    if cfg.dataset.name == 'NASL':
        cfg.dataset.num_classes = nasl_train_loader(cfg).dataset.num_classes
        return nasl_train_loader(cfg), nasl_validation_loader(cfg), nasl_test_loader(cfg)
    if cfg.dataset.name == 'NCARS':
        cfg.dataset.num_classes = ncars_train_loader(cfg).dataset.num_classes
        return ncars_train_loader(cfg), ncars_validation_loader(cfg), ncars_test_loader(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name}")