import time
import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.fabric.utilities import measure_flops

import event_dataset, models


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--test_only", action="store_true", default=False)
        parser.add_argument("--resume", type=str, default=None)
        '''
        example of resuming training
        --resume /home/wf282/project/event2vec/asl_dvs/checkpoints/version_0/last.ckpt --trainer.logger.class_path=TensorBoardLogger --trainer.logger.init_args.version=0 --trainer.logger.init_args.save_dir=./asl_dvs
        '''

def main():
    cli = CustomLightningCLI(models.Event2VecClassifier, event_dataset.EventDataModule, run=False)
    if cli.config['test_only']:
        cli.trainer.validate(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config['resume'])
        return


    # estimate_epochs(cli)
    cli.trainer.num_sanity_val_steps = 0

    ckp_path = os.path.join(cli.trainer.default_root_dir, "checkpoints",
                            os.path.split(cli.trainer.logger.log_dir)[1])
    save_ckp = lightning.pytorch.callbacks.ModelCheckpoint(dirpath=ckp_path,
                                                           filename="model-{epoch:02d}-{train_acc:.2f}-{valid_acc:.2f}",
                                                           save_last=True)
    cli.trainer.callbacks = [save_ckp, SaveConfigCallback(cli._parser(cli.subcommand),
                                                          cli.config.get(str(cli.subcommand), cli.config),
                                                          **cli.save_config_kwargs, )]  # note that it will remove the original default callbacks

    if cli.config['resume'] is not None:
        for cb in cli.trainer.callbacks:
            if isinstance(cb, SaveConfigCallback):
                cb.already_saved = True

    cli.model.print_info = str(cli.config)
    model = cli.model
    if model.compile_flag:
        t_start = time.time()
        model = torch.compile(model)
        print(f'compile time: {time.time() - t_start:.3f}s')

    cli.trainer.fit(model, datamodule=cli.datamodule, ckpt_path=cli.config['resume'])



if __name__ == '__main__':
    main()