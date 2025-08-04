from .dataloader_comsX2 import load_data as load_comsX2

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'comsX2': 
        return load_comsX2(batch_size, val_batch_size, data_root, num_workers, kwargs['aug_n'], kwargs['n_train'], kwargs['n_test'])
