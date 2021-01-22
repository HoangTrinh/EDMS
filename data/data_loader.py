def CreateDataLoader_stage1(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader_stage1
    data_loader = CustomDatasetDataLoader_stage1()
    data_loader.initialize(opt)
    return data_loader

def CreateDataLoader_stage2(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader_stage2
    data_loader = CustomDatasetDataLoader_stage2()
    data_loader.initialize(opt)
    return data_loader
