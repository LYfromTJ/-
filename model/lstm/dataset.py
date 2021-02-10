import random

from torch.utils.data import Dataset


class fileDataset(Dataset):
    def __init__(self, root, lSeq=5):
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        file_path = self.lines[index]
        # TODO: load dataset
        seq, label = load_data(file_path)
        return seq, label
