"""Shared dataset and DataLoader construction, following TSLib's data_provider entry."""

from torch.utils.data import DataLoader

from .data_loader import Dataset_MMSP


def data_provider(args, flag):
    if args.data != 'MMSP':
        raise ValueError('This project currently supports the migrated MMSP dataset')
    dataset = Dataset_MMSP(
        root_path=args.root_path, flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        protocol=args.protocol, modalities=args.modalities,
    )
    train = flag == 'train'
    loader = DataLoader(
        dataset, batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train, drop_last=False, num_workers=args.num_workers,
        pin_memory=args.use_gpu,
    )
    return dataset, loader
