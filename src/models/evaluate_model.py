# -*- coding: utf-8 -*-
"""Function to evaluate any model"""
from time import time

def evaluate_model(model, dataset, options):
    """Function to evaluate any model

    Arguments
    ---------
    model: BaseModel
        Model to use for evaluation
    dataset
        Dataset to use for evaluation
    options: BaseOptions
        Options to use while evaluating
    """
    # set model in evaluation mode
    model.eval()

    start_time = time()
    times = []
    for data_index, data in enumerate(dataset):# loop over data batches
        evaluation_start_time = time()
        if options.direction == 'both':
            model.evaluate(None, *data)
        else:
            model.evaluate(None, data, None)

        if options.verbose:
            evaluation_time = time() - evaluation_start_time
            times.append(evaluation_time)
            print(f"[{(data_index + 1)}/{len(dataset)}] - " +
                  f"time: {evaluation_time:.2f}")

    if options.verbose:
        total_eval_time = time() - start_time
        print("Avg one evaluation time: " +
              f"{((sum(times) / len(times)) if times else 0):.2f}, " +
              f"total {len(dataset)} evaluations time: {total_eval_time:.2f}")
