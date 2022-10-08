#! /usr/bin/env python
from torch.multiprocessing import set_start_method
from src.options import TrainingOptions
from src.data import CustomDatasetDataLoader
from src.models import create_model, train_model

if __name__ == "__main__":
    set_start_method('spawn')
    options = TrainingOptions()
    options.parse()

    dataset = CustomDatasetDataLoader(options.options)

    model = create_model(options)
    if options.options.verbose:
        print("-----------------Networks---------------")
        print(model)
        print("---------------End networks-------------")

    train_model(model, dataset, options.options)
