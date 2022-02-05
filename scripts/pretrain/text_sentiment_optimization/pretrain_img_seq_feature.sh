# hyper-params
LR=3e-4
DROPOUT=0.3
LAYER=3
WARMUP=6000
export CUDA_VISIBLE_DEVICES=0
# directory to save models
MODEL_DIR="../../../../models/2/"
PRETRAINED_DIR=$MODEL_DIR"/with_pretrain_layer${LAYER}_lr${LR}_bsz128_drop${DROPOUT}_warmup${WARMUP}"
PRETRAIN_DATA_DIR="../../../../data/preprocessed_data/"


PRETRAINED_MODEL_DIR=$PRETRAINED_DIR"/checkpoint_best.pt"
FINETUNE_DATA_DIR=""
SENTIMENT_PREDICTOR_DIR="D:\LM\bert-base-cased\pytorch_version"
TYPE="features"
video_dialogue_model="../../../video_dialogue_model"

fairseq-train \
  --save-dir $PRETRAINED_DIR \
  --user-dir $video_dialogue_model \
  --task tso \
  --tso \
  --bpe "gpt2" \
  --sentiment-predictor-dir $SENTIMENT_PREDICTOR_DIR \
  --data-dir $PRETRAIN_DATA_DIR \
  --arch baseline-img-tso-bart \
  --encoder-layers $LAYER \
  --decoder-layers $LAYER \
  --encoder-embed-dim 512 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 8000 \
  --batch-size 1 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $LR \
  --img-dim 1000 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 100 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --log-format json \
  --num-workers 0 \
  --sentence-avg \
  --criterion tso_loss \

#exit

FINETUNE_LR=3e-4
DROPOUT=0.3
FINETUNE_LAYER=3
WARMUP=6000

FINETUNE_MODEL_DIR=$PRETRAINED_DIR"/layer${FINETUNE_LAYER}_lr${FINETUNE_LR}_bsz128_drop${DROPOUT}_warmup${WARMUP}"


#fairseq-train \
  --tso \
  --bpe "gpt2" \
  --restore-file $PRETRAINED_MODEL_DIR \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir $FINETUNE_MODEL_DIR \
  --user-dir $video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $FINETUNE_DATA_DIR \
  --arch baseline-img-bart\
  --encoder-layers $FINETUNE_LAYER \
  --decoder-layers $FINETUNE_LAYER \
  --encoder-embed-dim 512 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 8000 \
  --batch-size 128 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $FINETUNE_LR \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 200 \
  --num-workers 0 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --use-img
exit
  #--input-img-dim 1000
# generate system predictions to OUTPUT
MODEL_PATH="${FINETUNE_MODEL_DIR}/checkpoint_best.pt"
OUTPUT="${FINETUNE_MODEL_DIR}/gen.out"
#python /OpenViDial/repo/train/generate.py \
  --user-dir $video_dialogue_model \
  --num-workers 0 \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $FINETUNE_DATA_DIR \
  --path $MODEL_PATH \
  --beam 5 \
  --batch-size 32 \
  --remove-bpe \
  --results-path $FINETUNE_MODEL_DIR \

  #--gen-subset "test" >$OUTPUT 2>&1 & tail -f $OUTPUT 2>&1 \
  
