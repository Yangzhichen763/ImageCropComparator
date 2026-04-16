
import torch.utils.data as data
import torchvision.transforms


class TransformsDatasetBase(data.Dataset):
    def __init__(self, **option):
        super(TransformsDatasetBase, self).__init__()

        if 'transforms' in option:
            self.transforms = option['transforms']
        else:
            self.transforms = None

