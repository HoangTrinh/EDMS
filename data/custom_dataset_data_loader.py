import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset_stage1(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader_stage1(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader_stage1'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_stage1(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataset_stage2(opt):
    dataset = None
    from data.comp_dataset import CompDataset
    dataset = CompDataset()
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader_stage2(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader_stage2'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_stage2(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
