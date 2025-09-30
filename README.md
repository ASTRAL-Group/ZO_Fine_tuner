# Learning a Zeroth-Order Optimizer for Fine-Tuning LLM
![Methodology](figures/methodology.png)

In the foundation-model era, most downstream models derive from a small set of base checkpoints, making it efficient to learn an optimizer once per base model and reuse it across many tasks. Moreover, doing so in the zeroth-order setting preserves near-inference compute and memory costs, greatly improving the accessibility of model customization. To this end, we propose ZO Fine-tuner, a learning-based zeroth-order optimizer that augments the standard two-point ZO update with learned, adaptive, and non-uniform perturbations. Guided by our theoretical analysis, ZO Fine-tuner achieves minimal memory overhead by sharing a single perturbation variance within each parameter group. Trained once under a learning-to-learn objective for a given base LLM, the same finetuner transfers across diverse tasks and derivative checkpoints, enabling a practical “train once, reuse widely” workflow. Experiments on 4 LLMs ranging from 1B to 30B and 7 diverse datasets show that ZO Fine-tuner outperforms prior zeroth-order baselines in 23/28 of the task-model combinations in terms of final loss and achieves an average of 2.5% improvement in accuracy compared to MeZO, thereby demonstrating strong performance and scalability for efficient LLM fine-tuning.


## Installation
```bash
conda create -n ZO_fine_tuner python==3.9.19
conda activate ZO_fine_tuner 
pip install -r requirements.txt
```

This environment can support the OPT, LLaMA, Qwen and other recent LLMs.
## Usage

Use `run_l2l.py` to learn a ZO Fine-tuner and use `run_zo_fune_tuner.py` for performing downstream zeroth-order fine-tuning with the learned optimizer.
```bash
python run_l2l.py {ARGUMENTS}
python run_zo_fune_tuner.py {ARGUMENTS}
```

We provide example scripts below for reproducing our experiments, for both learning the ZO Fine-Tuner and fine-tuning on a downstream task, respectively.
```bash
#learning to learn
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 NEED_NORMALIZATION=True LOAD_FLOAT_16=False LR_MLP=0.1 EPOCH=15  LR_UPDATE=1e-6 TRAIN_MODE='l2l' EPOCHS_PER_RESTART=5 MODEL=meta-llama/Llama-3.2-1B TASK=Copa LR_LLM=0.01  SAVE_MLP_PATH='./learned_finetuner/llama1B_finetuner.pth' bash ./scripts/l2l.sh

# zero-order fine-tuning with the learned optimizer
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 LR=1e-7 LOAD_FLOAT_16=True NEED_NORMALIZATION=True TRAIN_MODE='zo_fine_tuner' STEPS=20000 MODEL=meta-llama/Llama-3.2-1B TASK=SST2 MODE=ft LOAD_MLP_PATH='./learned_finetuner/llama1B_finetuner.pth' bash ./scripts/zo_fine_tuner.sh
```

