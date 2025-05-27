import models


def factory(cfg):
    if cfg.model.name == 'EST':
        return models.NetEST(cfg)            
    else:
        raise NotImplementedError(f'Model {cfg.model.name} not implemented')
