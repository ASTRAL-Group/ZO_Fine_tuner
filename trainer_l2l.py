import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np
import pdb
import wandb
from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
#from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
#from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers
import torch.optim as optim
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
import pdb
import torch.nn.functional as F
import wandb
import copy

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl



if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer_l2l(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:#T
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):#F
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self.do_grad_scaling = False#self.args.do_grad_scaling
        self.best_eval_loss = 100.0

        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        #pdb.set_trace()
        if has_length(train_dataloader):#T
            len_dataloader = len(train_dataloader) #4
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0: ###args.max_steps=-1
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)#64 = 16 * 4
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = False
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")


            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        
        self.data_type = None
        self.data_device = None
        self.names_of_parameter_to_optim = []
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param)) 
                self.names_of_parameter_to_optim.append(name)
        self.data_device = param.data.device
        self.data_type = param.data.dtype



        self.perturbation_variance = {}
        self.MLP_perturb = {}
        for name, param in model.named_parameters():
            if len(param.shape)==1:
                param_shape = param.shape[0]
            else:
                param_shape = param.shape[0] * param.shape[1]
            if param.requires_grad:
                self.MLP_perturb[name] = nn.Sequential(
                            nn.Linear(5, 64, device = param.data.device, dtype=param.data.dtype), 
                            nn.Tanh(), 
                            nn.Linear(64, 1, device = param.data.device, dtype=param.data.dtype))
                if 'bias' in name or 'norm' in name:
                    self.perturbation_variance[name] = torch.ones_like(param[0:1], device=param.data.device, dtype=param.data.dtype)
                else:
                    self.perturbation_variance[name] = torch.ones_like(param[0:1, 0:1], device=param.data.device, dtype=param.data.dtype)



        self.MLP_optim_params = []
        for layer in self.MLP_perturb.values():
            self.MLP_optim_params.extend(list(layer.parameters()))
        self.MLP_optimizer = optim.SGD(self.MLP_optim_params, lr=self.args.lr_mlp)
        self.history_loss = torch.tensor([0,0],dtype=self.data_type, device=self.data_device)

        learning_rate_end = 1e-8
        self.steps_per_epoch = len(train_dataloader) 
        self.restart_total_steps = self.steps_per_epoch * self.args.epochs_per_restart 
        param_dict = dict(model.named_parameters())
        self.model_params = [param for name, param in param_dict.items()]

        self.valid_gradient_num_param = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'embed' in name:
                    self.valid_gradient_num_param += param.shape[1] / 1e6 * 15 * self.args.train_batch_size * self.args.world_size
                elif param.dim() == 1:
                    self.valid_gradient_num_param += param.numel() / 1e6
                elif param.dim() == 2:
                    self.valid_gradient_num_param += param.shape[0] / 1e6 * param.shape[1]

        def linear_decay_lr(current_step, epochs_per_restart):
            decay_ratio = current_step / epochs_per_restart
            return max(0.0, 1 - decay_ratio) 
        self.LLM_optimizer = optim.SGD(self.model_params, lr=self.args.lr_llm, momentum=0.8)
        
        self.LLM_lr_scheduler = LambdaLR(
                    self.LLM_optimizer, lr_lambda=lambda step: linear_decay_lr(step, self.restart_total_steps))

        

        #for restart
        self.copy_model_for_restart = copy.deepcopy(model)
        for name, param in self.copy_model_for_restart.named_parameters():
            param.data = param.data.cpu()
        if self.args.model_name == "meta-llama/Llama-3.2-1B" or self.args.model_name == "facebook/opt-125m": 
            self.copy_model_for_restart.lm_head.weight = self.copy_model_for_restart.model.embed_tokens.weight.cpu()
        if self.args.model_name == "facebook/opt-30b":
            self.copy_model_for_restart.lm_head.weight = self.copy_model_for_restart.model.decoder.embed_tokens.weight.cpu()

        torch.cuda.empty_cache()

        self.meta_model_estimate = copy.deepcopy(model)
        if self.args.model_name == "meta-llama/Llama-3.2-1B" or self.args.model_name == "facebook/opt-125m": 
            self.meta_model_estimate.lm_head.weight = self.meta_model_estimate.model.embed_tokens.weight#.data.contiguous()
        if self.args.model_name == "facebook/opt-30b":
            self.meta_model_estimate.lm_head.weight = self.meta_model_estimate.model.decoder.embed_tokens.weight


        self.log_original_loss = []
        self.log_meta_loss = []
        self.true_gradient = {}
        self.cos_similarity = 0.0
        

        for epoch in range(epochs_trained, num_train_epochs):

            # ############ restart
            if epoch %self.args.epochs_per_restart ==0 and epoch != 0: 
                for (name, param_copy), (_, param) in zip(self.copy_model_for_restart.named_parameters(),model.named_parameters()):
                    param.data = param_copy.data.to(param.device)
                if self.args.model_name == "meta-llama/Llama-3.2-1B" or self.args.model_name == "facebook/opt-125m": 
                    model.lm_head.weight = model.model.embed_tokens.weight#.data.contiguous()
                if self.args.model_name == "facebook/opt-30b":
                    model.lm_head.weight = model.model.decoder.embed_tokens.weight
                

                param_dict = dict(model.named_parameters())
                self.model_params = [param for name, param in param_dict.items()]
                self.LLM_optimizer = optim.SGD(self.model_params, lr=self.args.lr_llm, momentum=0.8)
                self.LLM_lr_scheduler = LambdaLR(
                    self.LLM_optimizer, lr_lambda=lambda step: linear_decay_lr(step, self.restart_total_steps)
                )


            ######## Skip reading
            epoch_iterator = train_dataloader
            if args.past_index >= 0:
                self._past = None
            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps)
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)
            step = -1
            ######## Skip reading

            for step, inputs in enumerate(epoch_iterator):

                ######## Skip reading
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                if step % args.gradient_accumulation_steps == 0:
                   self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                ######## Skip reading
                for i in range(1): # main model update nums
                    for (name, param) in model.named_parameters():

                        param_obj_estimate, param_type = self.get_parameter_value(self.meta_model_estimate, name)
                        param_obj_estimate._parameters[param_type] = torch.nn.Parameter(param.data)   #clone() may be removed to save storage
                    
                    if self.args.model_name == "meta-llama/Llama-3.2-1B" or self.args.model_name == "facebook/opt-125m": 
                        self.meta_model_estimate.lm_head.weight.data = self.meta_model_estimate.model.embed_tokens.weight.data.contiguous()
                    if self.args.model_name == "facebook/opt-30b":
                        self.meta_model_estimate.lm_head.weight = self.meta_model_estimate.model.decoder.embed_tokens.weight
                    self.random_seed = np.random.randint(1000000000)
                    
                    if self.state.global_step == 0:
                        self.history_loss[:] = 0

                    self.generate_perturbation(model)
                    self.meta_perturb_parameters(model, scaling_factor=1)
                    self.loss_positive = self.zo_forward(model, inputs)
                    self.meta_perturb_parameters(model, scaling_factor=-2)
                    self.loss_negative = self.zo_forward(model, inputs)
                    self.meta_perturb_parameters(model, scaling_factor=1)
                    self.use_estimate_gradient_to_perturb(model)
                    
                    self.loss_estimate = self.meta_forward(self.meta_model_estimate, inputs)
                    ########smooth loss
                    self.log_original_loss.append(self.loss_positive.item())
                    self.log_meta_loss.append(self.loss_estimate.item())
                    if step % 10 == 0 and step != 0 :
                        lr = self.LLM_optimizer.param_groups[0]["lr"]
                        logger.info(f" Adamw_LR = {lr}")
                        original_loss = round(np.mean(self.log_original_loss), 4)
                        meta_loss = round(np.mean(self.log_meta_loss), 4)
                        logger.info(f" original_loss = {original_loss}")
                        logger.info(f" meta_loss = {meta_loss}")
                        self.log_original_loss = []
                        self.log_meta_loss = []
                        
                    #update the history loss
                    self.history_loss = torch.cat((self.loss_positive.detach().unsqueeze(0), 
                                               self.loss_negative.detach().unsqueeze(0)),0)

                    self.loss_estimate.backward()
                    self.MLP_optimizer.step()
                    self.MLP_optimizer.zero_grad()
                    self.cos_similarity = 0.0

                
                ## update LLM using Adam
                model.zero_grad()
                model.train()
                if True:  # add for loop for multi dataset
                    if (((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation): #F
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else: #T   ############place it in the last position
                        tr_loss_step = self.training_step(model, inputs) 
                    

                
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch): #T
                    # MeZO added: update model with the estimated gradient
                    if True:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping
                            if  self.do_grad_scaling:#T
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,)
                        
                        # Optimizer step
                        optimizer_was_run = True
                        if  self.do_grad_scaling:#T ###FP16
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else: #####FP32
                            self.LLM_optimizer.step()
                            self.LLM_lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if self.control.should_training_stop:
                break
        
        torch.save(self.MLP_perturb, self.args.save_mlp_path)

        ######## Skip reading
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        self._load_best_model()
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    def generate_perturbation(self, model):
        torch.manual_seed(self.random_seed)
        if self.args.need_normalization == True:
            self.normalization_norm = 0
        for (name, param) in self.named_parameters_to_optim:
            if self.args.need_normalization == True and isinstance(self.normalization_norm, torch.Tensor) and self.normalization_norm.device != param.device:
                self.normalization_norm = self.normalization_norm.to(param.device)
            if 'bias' in name or 'norm' in name:
                with torch.no_grad():
                    param_mean = torch.mean(param).unsqueeze(0)
                    param_var = torch.var(param).unsqueeze(0)
                    loss_state = self.history_loss.to(param.device)
                    model_state_temp = torch.cat((param_mean, param_var, loss_state, self.perturbation_variance[name].detach()))
                self.perturbation_variance[name] = torch.abs(self.MLP_perturb[name](model_state_temp.detach()))
                if self.args.need_normalization == True:
                    self.normalization_norm += param.numel() / 1e6 * self.perturbation_variance[name]**2
            else:
                with torch.no_grad():
                    param_mean = torch.mean(param).unsqueeze(0).unsqueeze(1)
                    param_var = torch.var(param).unsqueeze(0).unsqueeze(1)
                    loss_state = self.history_loss.unsqueeze(0).to(param.device)
                    model_state_temp = torch.cat((param_mean, param_var, loss_state, self.perturbation_variance[name].detach()), dim=1)
                self.perturbation_variance[name] = torch.abs(self.MLP_perturb[name](model_state_temp.detach()))
                if self.args.need_normalization == True:
                    if 'embed' in name:
                        self.normalization_norm += 15 * self.args.train_batch_size * self.args.world_size / 1e6 * param.shape[1] * self.perturbation_variance[name]**2
                    else:
                        self.normalization_norm +=  param.shape[0] / 1e6 * param.shape[1] * self.perturbation_variance[name]**2
        if self.args.need_normalization:        
            self.normalization_factor = torch.sqrt(self.valid_gradient_num_param) / torch.sqrt(self.normalization_norm).squeeze()        

    def activate_var(self, v):
        var = nn.Parameter(v.data, requires_grad=True)
        var.retain_grad()
        return var

    def meta_forward(self, model, inputs):
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            # Warning: this is copied from the original Huggingface Trainer. Untested.
            loss = loss.mean() 
        return loss

    def use_estimate_sample_to_perturb(self, model):
        torch.manual_seed(self.random_seed)
        for (name, param), (_, meta_positive_param), (_, meta_negative_param) in zip(model.named_parameters(),
                self.meta_model_positive.named_parameters(), self.meta_model_negative.named_parameters()):
            if self.args.need_normalization and self.normalization_factor.device != param.device:
                self.normalization_factor = self.normalization_factor.to(param.device)
            u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            perturbation_of_estimate_sample = (u * self.args.zo_eps).detach() * self.perturbation_variance[name]
            if self.args.need_normalization:
                perturbation_of_estimate_sample = perturbation_of_estimate_sample * self.normalization_factor.item()

            param_obj_positive, param_type = self.get_parameter_value(self.meta_model_positive, name)
            param_obj_positive._parameters[param_type] = meta_positive_param.detach() + perturbation_of_estimate_sample
            
            param_obj_negative, param_type = self.get_parameter_value(self.meta_model_negative, name)
            param_obj_negative._parameters[param_type] = meta_negative_param.detach() - perturbation_of_estimate_sample

        
        if self.args.model_name == "meta-llama/Llama-3.2-1B" or self.args.model_name == "facebook/opt-125m":     
            self.meta_model_positive.lm_head._parameters['weight'] = self.meta_model_positive.model.embed_tokens.weight
            self.meta_model_negative.lm_head._parameters['weight'] = self.meta_model_negative.model.embed_tokens.weight
        if self.args.model_name == "facebook/opt-30b":
            self.meta_model_positive.lm_head._parameters['weight'] = self.meta_model_positive.model.decoder.embed_tokens.weight
            self.meta_model_negative.lm_head._parameters['weight'] = self.meta_model_negative.model.decoder.embed_tokens.weight


    def use_estimate_gradient_to_perturb(self, model):
        torch.manual_seed(self.random_seed)
        for (name, param), (_, meta_estimate_param) in zip(model.named_parameters(),
                self.meta_model_estimate.named_parameters()):
            if self.args.need_normalization and self.normalization_factor.device != param.device:
                self.normalization_factor = self.normalization_factor.to(param.device)
            loss_scale = (self.loss_positive.to(param.device)-self.loss_negative.to(param.device)) / self.args.zo_eps / 2
            u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param_obj_estimate, param_type = self.get_parameter_value(self.meta_model_estimate, name)
            estimate_gradient = (-1 * u * loss_scale).detach() * self.perturbation_variance[name]
            if self.args.need_normalization:
                estimate_gradient = estimate_gradient * self.normalization_factor.item()
            param_obj_estimate._parameters[param_type] = meta_estimate_param.detach() + estimate_gradient * self.args.lr_update
            
        if self.args.model_name == "meta-llama/Llama-3.2-1B" or self.args.model_name == "facebook/opt-125m":        
            self.meta_model_estimate.lm_head._parameters['weight'] = self.meta_model_estimate.model.embed_tokens.weight
        if self.args.model_name == "facebook/opt-30b":
            self.meta_model_estimate.lm_head._parameters['weight'] = self.meta_model_estimate.model.decoder.embed_tokens.weight
        

    def get_parameter_value(self, model, param_name):
        """
        :param_name:  'model.decoder.final_layer_norm.weight'
        :return: make it possible to access by:  model.model.decoder.final_layer_norm._parameters['weight']
        """
        attributes = param_name.split('.')
        param_obj = model
        for attr in attributes[:-1]:  
            param_obj = getattr(param_obj, attr)
        param_type = attributes[-1]
        return param_obj, param_type

    def meta_perturb_parameters(self, model, scaling_factor=1):
        torch.manual_seed(self.random_seed)
        for name, param in self.named_parameters_to_optim:
            with torch.no_grad():
                if self.args.need_normalization and self.normalization_factor.device != param.device:
                    self.normalization_factor = self.normalization_factor.to(param.device)
                u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if self.args.need_normalization:
                    param.data = param.data + scaling_factor * u * self.perturbation_variance[name]  * self.normalization_factor.item() * self.args.zo_eps
                else:
                    param.data = param.data + scaling_factor * u * self.perturbation_variance[name] * self.args.zo_eps
    def zo_forward(self, model, inputs):
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()
    
    ########no need to read below

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None: #F
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        if self.state.global_step%10==0: #self.control.should_log:#
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["train_loss"] = round(tr_loss_scalar / (10), 4)
            logs["learning_rate"] = self._get_learning_rate()


            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.state.global_step%self.args.save_steps==0:#self.control.should_evaluate:#
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        
            if metrics['eval_loss']<self.best_eval_loss:# self.control.should_save:
                self.control.should_save = True
                self._save_checkpoint(model, trial, metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                self.control.should_save = False


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)


        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
