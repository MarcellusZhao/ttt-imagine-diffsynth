import os, math, torch, importlib
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def create_lr_scheduler(optimizer, lr_scheduler="constant", total_steps=None, warmup_steps=30, min_ratio=0.1):
    if lr_scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif lr_scheduler == "cosine_warmup":
        warmup_steps = max(1, warmup_steps)
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            if total_steps is None or total_steps <= warmup_steps:
                return 1.0
            progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_scheduler `{lr_scheduler}`. Supported: `constant`, `cosine_warmup`.")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 30,
    lr_min_ratio: float = 0.1,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        # getattr: direct callers may pass args objects predating these flags.
        lr_scheduler = getattr(args, "lr_scheduler", lr_scheduler)
        lr_warmup_steps = getattr(args, "lr_warmup_steps", lr_warmup_steps)
        lr_min_ratio = getattr(args, "lr_min_ratio", lr_min_ratio)
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Built after prepare (and kept out of it) so the schedule advances exactly once per
    # optimizer step on every rank: the prepared dataloader length is per-rank, and an
    # accelerate-prepared scheduler would tick num_processes times per step instead.
    total_steps = num_epochs * math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
    scheduler = create_lr_scheduler(optimizer, lr_scheduler, total_steps, lr_warmup_steps, lr_min_ratio)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                # Optional global grad-norm logging: only once grads are synced (accum-correct)
                # and only for modules that opt in via `log_grad_norm`. Stashed into the
                # module's `log_metrics` so ModelLogger.on_step_end picks it up alongside the
                # forward-time diagnostics. No-op (and untouched behaviour) for every other family.
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    if getattr(unwrapped, "log_grad_norm", False):
                        total_sq = None
                        for p in unwrapped.trainable_modules():
                            if p.grad is not None:
                                g = p.grad.detach().float().pow(2).sum()
                                total_sq = g if total_sq is None else total_sq + g
                        if total_sq is not None:
                            if getattr(unwrapped, "log_metrics", None) is None:
                                unwrapped.log_metrics = {}
                            key = getattr(unwrapped, "grad_norm_log_key", "train/grad_norm")
                            unwrapped.log_metrics[key] = total_sq.sqrt()
                optimizer.step()
                # Advance only on real optimizer steps (accumulation micro-batches skip).
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, lr=optimizer.param_groups[0]["lr"])
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
