import torch

def create_model(opt):
    from .EDMS_model import EDMSModel
    model = EDMSModel()
    model.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
