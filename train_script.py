import torch
import triton
import inspect

# 1. 保存原始的 triton.autotune 函数
original_autotune = triton.autotune


# 2. 定义新的、更健壮的装饰器工厂
def patched_autotune(*args, **kwargs):
    """
    这是一个装饰器工厂。它接收@triton.autotune()的参数，
    并返回一个真正的装饰器。
    """

    # 3. 定义真正的装饰器，它将被应用到内核函数上
    def decorator(fn):
        """这个函数是真正的装饰器，它包裹了内核函数 fn。"""

        # 安全地从 args 或 kwargs 中提取 'configs' 列表
        configs = kwargs.get('configs')
        if configs is None:
            if not args:
                # 如果既不在kwargs也不在args，无法操作，恢复原始行为
                # print("-> 警告：未找到 'configs' 参数。恢复原始 autotune 行为。")
                return original_autotune(*args, **kwargs)(fn)
            configs = args[0]

        # print("猴子补丁已生效：Triton Autotuner 将强制使用第一个配置。")

        # 如果配置列表为空，也恢复原始行为
        if not configs:
            # print("-> 警告：配置列表为空。恢复原始 autotune 行为。")
            return original_autotune(*args, **kwargs)(fn)

        # 强制选择第一个配置
        first_config = configs[0]
        # print(f"-> 原始配置数量: {len(configs)}")
        # print(f"-> 强制选择的配置: {first_config}")

        # 创建新的参数来调用原始的 autotune。
        # 优先修改 kwargs，因为它更明确。
        new_kwargs = kwargs.copy()
        new_kwargs['configs'] = [first_config]

        # 如果 configs 原本是在 args 里的，我们需要构建一个新的 args 元组
        if 'configs' not in kwargs and args:
            new_args = ([first_config],) + args[1:]
            # 使用新的参数调用原始的autotune工厂，然后将返回的装饰器应用到函数fn上
            return original_autotune(*new_args, **new_kwargs)(fn)
        else:
            # 如果configs本来就在kwargs里，或者没有其他位置参数，直接用new_kwargs
            return original_autotune(**new_kwargs)(fn)

    # 4. 返回这个准备好的装饰器
    return decorator


# 5. 应用猴子补丁
triton.autotune = patched_autotune

import time
import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.fabric.utilities import measure_flops
import torch.distributed as dist

import event_dataset, models


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--test_only", action="store_true", default=False)
        parser.add_argument("--resume", type=str, default=None)
        '''
        example of resuming training
        --resume /home/wf282/project/event2vec/asl_dvs/checkpoints/version_0/last.ckpt --trainer.logger.class_path=TensorBoardLogger --trainer.logger.init_args.version=0 --trainer.logger.init_args.save_dir=./asl_dvs
        '''



def set_numa_binding(numa_node):
    """为当前进程设置 NUMA 绑定"""
    cmd = f"numactl --membind={numa_node} --cpunodebind={numa_node} --"
    os.environ["NUMACTL_CMD"] = cmd
    print(cmd)
    # 或者直接调用 numactl（需要重新启动进程，较复杂）
    # subprocess.run(f"numactl --membind={numa_node} --cpunodebind={numa_node} ...", shell=True)


def main():
    cli = CustomLightningCLI(models.Event2VecClassifier, event_dataset.EventDataModule, run=False)

    rank = cli.trainer.local_rank

    # GPU 和 NUMA 节点映射（根据 nvidia-smi topo -m 确认）
    numa_nodes = {0: 0, 1: 0, 2: 1, 3: 1}  # GPU 0-1 绑定到 NUMA 0，GPU 2-3 绑定到 NUMA 1

    # 设置 NUMA 绑定
    set_numa_binding(numa_nodes[rank])

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

    # check
    assert cli.datamodule.batch_size == cli.model.batch_size

    if cli.trainer.global_rank == 0:
        print(cli.model)
    cli.model.print_info = str(cli.config)
    model = cli.model
    if model.compile_flag:
        t_start = time.time()
        model = torch.compile(model)
        print(f'compile time: {time.time() - t_start:.3f}s')

    cli.trainer.fit(model, datamodule=cli.datamodule, ckpt_path=cli.config['resume'])
if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    main()