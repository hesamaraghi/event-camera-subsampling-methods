import os
os.environ['WANDB__SERVICE_WAIT'] = "1000"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
import glob 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import torchmetrics
import copy
import numpy as np
import time

# Imports of own files
import model_factory
from graph_data_module import GraphDataModule


class Runner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = eval(cfg.train.loss_fn)
        self.multi_val_num = cfg.train.multi_val_num if hasattr(cfg.train, 'multi_val_num') and cfg.train.multi_val_num is not None else 1
        self.multi_test_num = cfg.train.multi_test_num if hasattr(cfg.train, 'multi_test_num') and cfg.train.multi_test_num is not None else 1
        
        self.val_loss = torch.tensor([0.0 for _ in range(self.multi_val_num)])
        self.test_loss = torch.tensor([0.0 for _ in range(self.multi_test_num)])
        self.val_num_samples = torch.tensor([0 for _ in range(self.multi_val_num)])
        self.test_num_samples = torch.tensor([0 for _ in range(self.multi_test_num)])

        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)
        self.val_accuracy =  nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes) for _ in range(self.multi_val_num) ])
        self.test_accuracy = nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes) for _ in range(self.multi_test_num) ])

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual
        # network
        return self.model(x)

    def configure_optimizers(self):
        if self.cfg.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), 
                                         lr=self.cfg.optimize.lr,
                                         weight_decay=self.cfg.optimize.weight_decay if hasattr(self.cfg.optimize, 'weight_decay') and 
                                         self.cfg.optimize.weight_decay is not None else 0)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.optimizer}")
        
        if self.cfg.optimize.lr_scheduler == 'None':
            return optimizer
        elif self.cfg.optimize.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.optimize.T_max)
        elif self.cfg.optimize.lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.cfg.optimize.T_0)
        elif self.cfg.optimize.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.cfg.optimize.mode, factor=self.cfg.optimize.factor, patience=self.cfg.optimize.patience) 
        else:
            raise NotImplementedError(f"Scheduler {self.cfg.optimize.lr_scheduler}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc/mean",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def _step(self, batch):
        y = batch.y
        y_hat = self.model(batch)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy(preds, batch.y)

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=len(batch.label), sync_dist=True)
        self.log("train/acc_step", self.train_accuracy, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=len(batch.label), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        batch_size = torch.tensor(len(batch.label))
        self.val_accuracy[dataloader_idx](preds, batch.y)
        self.val_loss[dataloader_idx] += loss.to('cpu') * batch_size
        self.val_num_samples[dataloader_idx] += batch_size
        
        # Log step-level loss & accuracy
        self.log(f"val/loss_step", loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"val/acc_step", self.val_accuracy[dataloader_idx], on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        batch_size = torch.tensor(len(batch.label))
        self.test_accuracy[dataloader_idx](preds, batch.y)
        self.test_loss[dataloader_idx] += loss.to('cpu') * batch_size
        self.test_num_samples[dataloader_idx] += batch_size
        
        # Log test loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('test/acc', self.test_accuracy[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

    # def on_train_epoch_end(self):
    #     # Log the epoch-level training accuracy
    #     self.log('train/acc', self.train_accuracy.compute())
    #     self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Log the epoch-level validation accuracy
        acc_list = torch.tensor([metric.compute() for metric in self.val_accuracy])
        self.log('val/acc/mean', torch.mean(acc_list), sync_dist=True)
        self.log('val/acc/std', torch.std(acc_list), sync_dist=True)
        self.val_loss = self.val_loss / self.val_num_samples
        self.log('val/loss/mean', torch.mean(self.val_loss), sync_dist=True)
        self.log('val/loss/std', torch.std(self.val_loss), sync_dist=True)
        for metric in self.val_accuracy:
            metric.reset()
        self.val_loss = torch.tensor([0.0 for _ in range(self.multi_val_num)])
        self.val_num_samples = torch.tensor([0 for _ in range(self.multi_val_num)])
  
            
    def on_test_epoch_end(self):
        # Log the epoch-level validation accuracy
        acc_list = torch.tensor([metric.compute() for metric in self.test_accuracy])
        self.log('test/acc_mean', torch.mean(acc_list), sync_dist=True)
        self.log('test/acc_std', torch.std(acc_list), sync_dist=True)
        self.test_loss = self.test_loss / self.test_num_samples
        self.log('test/loss_mean', torch.mean(self.test_loss), sync_dist=True)
        self.log('test/loss_std', torch.std(self.test_loss), sync_dist=True)
        for metric in self.test_accuracy:
            metric.reset()
        self.test_loss = torch.tensor([0.0 for _ in range(self.multi_test_num)])
        self.test_num_samples = torch.tensor([0 for _ in range(self.multi_test_num)])

def main(cfg_path: str = None):
    # Load defaults and overwrite by command-line arguments

    cmd_cfg = OmegaConf.from_cli()
    
    if hasattr(cmd_cfg, 'cfg_path') and cmd_cfg.cfg_path is not None:
        cfg_path = cmd_cfg.cfg_path
        print(f"cfg_path is loaded from command line: cfg_path={cfg_path}")
    else:
        if not cfg_path:
            raise ValueError("cfg_path is not provided")
            
    cfg = OmegaConf.load(cfg_path)
    cfg.cfg_path = cfg_path
    
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    cfg_bare = OmegaConf.load("config_bare.yaml")
    cfg = OmegaConf.merge(cfg_bare,cfg)
    
    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, 'cache')
    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        id=cfg.wandb.id if hasattr(cfg.wandb, 'id') and cfg.wandb.id is not None else None,
        name=cfg.wandb.experiment_name,
        log_model=cfg.wandb.log,
        offline=cfg.wandb.offline if hasattr(cfg.wandb, 'offline') and 
                                         cfg.wandb.offline is not None else not cfg.wandb.log,
        # Keyword args passed to wandb.init()
        entity=cfg.wandb.entity,
        config=OmegaConf.to_object(cfg),
    )
    
    print(OmegaConf.to_yaml(cfg), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(cfg.seed, workers=True)

    # Create datasets using factory pattern
    
    start = time.time()
    gdm = GraphDataModule(cfg)
    print(f"Time to create GraphDataModule: {time.time()-start}", flush=True)
    cfg.dataset.num_classes = gdm.num_classes
    
    # Create model using factory pattern
    model = model_factory.factory(cfg)

    # Tie it all together with PyTorch Lightning: Runner contains the model,
    # optimizer, loss function and metrics; Trainer executes the
    # training/validation loops and model checkpointing.
    runner = Runner(cfg, model)
    
    callback_list = []
    callback_list.append(LearningRateMonitor(logging_interval='step'))
    callback_list.append(TQDMProgressBar(refresh_rate=50))
    callback_list.append(ModelCheckpoint())
    
    if hasattr(cfg.train, 'default_root_dir') and cfg.train.default_root_dir is not None:
        default_root_dir = cfg.train.default_root_dir
    else:
        default_root_dir = os.path.join(os.getcwd(),'pl_default_dir', wandb_logger.experiment.project, wandb_logger.experiment.name)
        if not os.path.exists(default_root_dir):
            os.makedirs(default_root_dir)
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        enable_progress_bar=True,
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
        devices=torch.cuda.device_count(),
        accelerator="auto",
        callbacks=callback_list,
        profiler=cfg.train.profiler,
        default_root_dir=default_root_dir
    )
    
    # torch.set_num_threads(1)
    print(f"Number of threads: {torch.get_num_threads()}")
    if cfg.dataset.num_workers > 0:
        num_to_set = min(cfg.dataset.num_workers, torch.get_num_threads() - 1)
        torch.set_num_threads(num_to_set)
        print(f"Number of threads is set to: {torch.get_num_threads()}")

    # Train + validate (if validation dataset is implemented)
    trainer.fit(model = runner, datamodule=gdm)

    # Test (if test dataset is implemented)
    if gdm.test_dataloader is not None:
        trainer.test(datamodule=gdm, ckpt_path='last')

    # delete the default_root_dir checkpoints
    if os.path.exists(default_root_dir):
        files_to_delete = glob.glob(os.path.join(default_root_dir, "hpc_ckpt_*.ckpt"))
        for file_path in files_to_delete:
            os.remove(file_path)
            
    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

        
if __name__ == '__main__':
    main()