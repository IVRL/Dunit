# -*- coding: utf-8 -*-
"""Function to train any model"""
import os
from time import time
import torch

def train_model(model, dataset, options):
    """Function to train any model

    Arguments
    ---------
    model: BaseModel
        Model to train
    dataset
        Dataset to use for training
    options: BaseOptions
        Options to use while training
    """
    # set model in training mode
    model.train()
    # pretrain model if needed
    model.pretrain(dataset)

    start_time = time()
    times = []
    for epoch in range(options.starting_epoch, options.nb_epochs):
        epoch_start_time = time()
        for data in dataset:# loop over data batches
            for optimizer in model.optimizers.values():
                optimizer.zero_grad()

            model.train_epoch(*data)
        model.update_learning_rates()

        # print log
        if options.verbose:
            epoch_time = time() - epoch_start_time
            times.append(epoch_time)
            print(
                f"[{(epoch + 1)}/{options.nb_epochs}] - time: " +
                f"{epoch_time:.2f} - {model.log_end_epoch(len(dataset))}")

        # save model
        if options.save and (
                (options.save_frequency > 0 and
                 epoch % options.save_frequency == options.save_frequency - 1)
                or epoch + 1 == options.nb_epochs):
            model.save(os.path.join(
                options.save_path, f"epoch_{(epoch + 1)}.pth"))

        # save examples
        if options.evaluate and (
                (options.evaluation_frequency > 0 and
                 epoch % options.evaluation_frequency ==
                 options.evaluation_frequency - 1)
                or epoch + 1 == options.nb_epochs):
            if options.verbose:
                print("Evaluating examples")
            with torch.no_grad():
                model.eval()
                for index, data in enumerate(dataset):
                    model.evaluate(epoch, *data)
                    if index == options.nb_evaluation_examples:
                        break
                model.train()

    if options.verbose:
        total_training_time = time() - start_time
        print(
            "Avg one epoch time: " +
            f"{((sum(times) / len(times)) if times else 0):.2f}, " +
            f"total {options.nb_epochs} epochs time: {total_training_time:.2f}")
