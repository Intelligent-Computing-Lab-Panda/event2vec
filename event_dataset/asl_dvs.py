import types
from .utils import EventNPDataset

class ASLDVS(EventNPDataset):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: types.FunctionType, repeats:int=1):
        super().__init__(training=training, root=root, sample_number=sample_number, sampler=sampler, repeats=repeats)

        train_ratio = 0.8
        num_classes = 24

        samples_in_class = []
        for i in range(num_classes):
            samples_in_class.append([])

        for i in range(len(self.samples)):
            path, label = self.samples[i]
            samples_in_class[label].append([path, label])

        self.samples.clear()
        for i in range(num_classes):
            pos = int(len(samples_in_class[i]) * train_ratio)
            if self.training:
                self.samples.extend(samples_in_class[i][0:pos])
            else:
                self.samples.extend(samples_in_class[i][pos:])

        del self.classes
        del self.class_to_idx
        del self.targets


    @staticmethod
    def num_classes():
        return 24


    @staticmethod
    def event_size():
        P = 2
        H = 180
        W = 240
        return P, H, W

