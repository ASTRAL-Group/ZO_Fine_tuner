MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

LR_UPDATE=${LR_UPDATE:-1e-6}
TRAIN_MODE=${TRAIN_MODE:-"l2l"}
LOAD_MLP_PATH=${LOAD_MLP_PATH:-"N"}
SAVE_MLP_PATH=${SAVE_MLP_PATH:-"N"}
LR_MLP=${LR_MLP:-1e-3}
EPOCHS_PER_RESTART=${EPOCHS_PER_RESTART:-4}
LOAD_FLOAT_16=${LOAD_FLOAT_16:-False}
NEED_NORMALIZATION=${NEED_NORMALIZATION:-False}
MULTI_TASK_NAME=${MULTI_TASK_NAME:-"SST2 Copa WSC"}
MULTI_TASK_TRAINING=${MULTI_TASK_TRAINING:-False}
SHUFFLE=${SHUFFLE:-False}
ZO_EPS=${ZO_EPS:-1e-3}
TASK=${TASK:-"Copa"}

EPOCH=${EPOCH:-20}
BS=${BS:-8}
LR_LLM=${LR_LLM:-1e-5}
LR=${LR_LLM:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-50}
EVAL=${EVAL:-1000}


TAG=$TASK-$EPOCH-$BS-$LR_LLM-$SEED-$LOAD_MLP_PATH-$SAVE_MLP_PATH

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    MultiRC) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR_MLP: $LR_MLP"
echo "LR_LLM: $LR_LLM"
echo "SEED: $SEED"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"


python3 run_l2l.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --trainer regular --multi_task_name $MULTI_TASK_NAME --multi_task_training $MULTI_TASK_TRAINING --zo_eps $ZO_EPS --per_device_train_batch_size $BS \
    --lr_llm $LR_LLM --learning_rate $LR --num_train_epochs $EPOCH --lr_update $LR_UPDATE --lr_mlp $LR_MLP --need_normalization $NEED_NORMALIZATION \
    --evaluation_strategy epoch --save_strategy no --save_total_limit 1 --shuffle $SHUFFLE \
    --train_as_classification --train_mode $TRAIN_MODE\
    --epochs_per_restart $EPOCHS_PER_RESTART --load_float16 $LOAD_FLOAT_16\
    --load_mlp_path $LOAD_MLP_PATH --save_mlp_path $SAVE_MLP_PATH\
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"

    # --load_best_model_at_end 
