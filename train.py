from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json
import pytorch_lightning as pl
import torch

# torch.multiprocessing.set_sharing_strategy("file_system")

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from transformercvn.options import Options


def main(
        log_dir: str,
        name: str,
        options_file: str,
        training_file: str,
        checkpoint: Optional[str],
        fp16: bool,
        graph: bool,
        verbose: bool,
        batch_size: Optional[int],
        gpus: Optional[int],
        threads: Optional[int],
        debug: bool,
        sparse: bool,
        sdxl: bool,
        eval: int,
        **kwargs
):
    master = "NODE_RANK" not in environ

    if sparse:
        from transformercvn.network.trainers.neutrino_full_sparse_trainer import NeutrinoFullSparseTrainer
        Network = NeutrinoFullSparseTrainer
    elif sdxl:
        from transformercvn.network.trainers.neutrino_full_sdxl_trainer import NeutrinoFullSDXLTrainer
        Network = NeutrinoFullSDXLTrainer
    else:
        from transformercvn.network.trainers.neutrino_full_dense_trainer import NeutrinoFullDenseTrainer
        Network = NeutrinoFullDenseTrainer

    options = Options()
    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # Apply Command line overrides for common option values.
    # ---------------------------------------------------------------------------------------------
    options.verbose_output = verbose
    if training_file is not None: options.training_file = training_file

    if threads is not None:
        if master:
            print(f"Setting CPU count: {threads}")

        torch.set_num_threads(threads)
        environ["OMP_NUM_THREADS"] = str(threads)
        environ["MKL_NUM_THREADS"] = str(threads)

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if debug:
        if master:
            print(f"Debug Mode: 1 GPU, 0 dataloader workers, Small Batch size")
        torch.jit.script = lambda x: x
        options.num_dataloader_workers = 0
        options.batch_size = 32

    # Print the full hyperparameter list
    # ---------------------------------------------------------------------------------------------
    if master:
        options.display()

    # Create the initial model on the CPU
    # ---------------------------------------------------------------------------------------------
    model = Network(options)

    # Create Loggers and Checkpoint systems
    # ---------------------------------------------------------------------------------------------
    if debug:
        logger = False
        callbacks = None

    else:
        # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
        log_dir = getcwd() if log_dir is None else log_dir
        logger = TensorBoardLogger(save_dir=log_dir, name=name, log_graph=graph)

        checkpoint_callback = ModelCheckpoint(
            verbose=options.verbose_output,
            every_n_train_steps=eval,
            monitor="val_epoch_AUC",
            mode="max",
            save_top_k=5,
            save_last=True
        )

        callbacks = [
            checkpoint_callback,
            LearningRateMonitor(),
            RichProgressBar(),
            RichModelSummary(max_depth=3)
        ]

    distributed_backend = None
    if options.num_gpu > 1:
        distributed_backend = DDPStrategy(
            # find_unused_parameters=False
        )

    # Create the final pytorch-lightning manager
    # ---------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=options.epochs,
        callbacks=callbacks,
        resume_from_checkpoint=checkpoint,
        strategy=distributed_backend,
        accelerator='gpu' if options.num_gpu > 0 else None,
        devices=options.num_gpu if options.num_gpu > 0 else None,
        track_grad_norm=2 if options.verbose_output else -1,
        gradient_clip_val=options.gradient_clip,
        precision=16 if fp16 else 32,
        val_check_interval=eval
    )

    if master and not debug:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)
        with open(trainer.logger.log_dir + "/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-t", "--training_file", type=str, default=None,
                        help="Input file containing training data.")

    parser.add_argument("-o", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load from")

    parser.add_argument("-n", "--name", type=str, default="lightning_logs",
                        help="The sub-directory to create for this run.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs.")

    parser.add_argument("-fp16", action="store_true",
                        help="Use AMP for training.")

    parser.add_argument("-g", "--graph", action="store_true",
                        help="Log the computation graph.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-e", "--eval", type=int, default=500,
                        help="Number of steps before eval")

    parser.add_argument("--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")

    parser.add_argument("--threads", type=int, default=None,
                        help="Override CPU count in hyperparameters.")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debug options super-switch. ")

    parser.add_argument("--sparse", action="store_true",
                        help="Use Sparse Network")

    parser.add_argument("--sdxl", action="store_true",
                        help="Use SDXL Network")

    main(**parser.parse_args().__dict__)
