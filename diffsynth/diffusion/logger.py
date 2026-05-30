import os, torch
from accelerate import Accelerator


class TensorBoardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard is enabled. Run `tensorboard --logdir={log_dir}` to visualize the training progress.")

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def log_dict(self, metrics, step):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class SwanLabLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None, run_name=None, config=None):
        import swanlab
        project_name = os.environ.get("SWANLAB_PROJECT", project_name)
        self.swanlab = swanlab
        self.swanlab.init(project=project_name, logdir=log_dir, experiment_name=run_name, config=config)
        print(f"SwanLab is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.swanlab.log({key: value}, step=step)

    def log_dict(self, metrics, step):
        self.swanlab.log(metrics, step=step)

    def close(self):
        self.swanlab.finish()


class WandbLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None, run_name=None, config=None, entity=None, mode=None):
        import wandb
        project_name = os.environ.get("WANDB_PROJECT", project_name)
        self.wandb = wandb
        # Default to OFFLINE so air-gapped HPC compute nodes never block on the network
        # during init; override with `export WANDB_MODE=online` (or `disabled`). setdefault
        # leaves any WANDB_MODE the user already exported untouched, and an explicit
        # `mode=` argument still wins over the env var.
        os.environ.setdefault("WANDB_MODE", "offline")
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        self.run = self.wandb.init(
            project=project_name, dir=log_dir, name=run_name,
            config=config, entity=entity, mode=mode,
        )
        print(f"Wandb is enabled. Project: {project_name}, run: {self.run.name}, mode: {os.environ.get('WANDB_MODE')}")

    def log(self, key, value, step):
        self.wandb.log({key: value}, step=step)

    def log_dict(self, metrics, step):
        if metrics:
            self.wandb.log(metrics, step=step)

    def close(self):
        self.wandb.finish()


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
        enable_tensorboard_log=False,
        enable_swanlab_log=False, swanlab_project="DiffSynth-Studio",
        enable_wandb_log=False, wandb_project="DiffSynth-Studio",
        wandb_run_name=None, wandb_entity=None, config=None,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # Loggers
        self.enable_tensorboard_log = enable_tensorboard_log
        self.enable_swanlab_log = enable_swanlab_log
        self.swanlab_project = swanlab_project
        self.enable_wandb_log = enable_wandb_log
        self.wandb_project = wandb_project
        # Run name defaults to the output dir basename (e.g. "Wan2.1-T2V-1.3B_e2e_ttt_smoke"),
        # so each output_path gets a self-describing wandb/swanlab run without extra flags.
        self.run_name = wandb_run_name or (os.path.basename(os.path.normpath(output_path)) if output_path else None)
        self.wandb_entity = wandb_entity
        self.config = config
        self.loggers = []
        self.loggers_initialized = False

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(TensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(SwanLabLogger(
                project_name=self.swanlab_project, log_dir=os.path.join(self.output_path, "swanlab_log"),
                run_name=self.run_name, config=self.config,
            ))
        if self.enable_wandb_log:
            self.loggers.append(WandbLogger(
                project_name=self.wandb_project, log_dir=os.path.join(self.output_path, "wandb_log"),
                run_name=self.run_name, config=self.config, entity=self.wandb_entity,
            ))
        self.loggers_initialized = True

    @staticmethod
    def _scalar(value):
        """Detach/sync a tensor to a python float so loggers never hold onto the graph."""
        return value.item() if torch.is_tensor(value) else value

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if accelerator.is_main_process:
            if not self.loggers_initialized:
                self.init_loggers()
            metrics = {}
            loss = kwargs.get("loss")
            if loss is not None:
                metrics["loss"] = self._scalar(loss)
            lr = kwargs.get("lr")
            if lr is not None:
                metrics["lr"] = self._scalar(lr)
            # Extra per-step diagnostics: an explicit `metrics=` kwarg, or a `log_metrics`
            # dict the training module sets on its `forward` (e.g. the E2E-TTT inner-loop
            # stats — memorize_loss, num_pred_pairs, inner_lr). Both are optional, so vanilla
            # SFT training keeps logging just loss/lr.
            extra = kwargs.get("metrics") or getattr(accelerator.unwrap_model(model), "log_metrics", None)
            if extra:
                for key, value in extra.items():
                    metrics[key] = self._scalar(value)
            if metrics:
                for logger in self.loggers:
                    logger.log_dict(metrics, self.num_steps)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        for logger in self.loggers:
            logger.close()

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
