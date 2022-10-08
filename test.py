#! /usr/bin/env python
from torch.multiprocessing import set_start_method
from src.options import TestingOptions
from src.data import CustomDatasetDataLoader
from src.models import load_model, evaluate_model

if __name__ == "__main__":
    set_start_method('spawn')
    options = TestingOptions()
    options.parse()


    model = load_model(options)
    dataset = CustomDatasetDataLoader(options.options)

    if options.options.verbose:
        print("-----------------Networks---------------")
        print(model)
        print("---------------End networks-------------")

    evaluate_model(model, dataset, options.options)
