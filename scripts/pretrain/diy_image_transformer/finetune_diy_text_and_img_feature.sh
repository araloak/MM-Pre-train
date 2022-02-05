# hyper-params
LR=3e-4
DROPOUT=0.3
LAYER=3
WARMUP=1
export CUDA_VISIBLE_DEVICES=0
# directory to save models
PRETRAINED_DIR="../../../models/2/"
MODEL_DIR=$PRETRAINED_DIR"/with_pretrain_layer${LAYER}_lr${LR}_bsz128_drop${DROPOUT}_warmup${WARMUP}"

PRETRAINED_MODEL_DIR=$PRETRAINED_DIR"/checkpoint_best.pt"
DATA_DIR=

TYPE="diy_features"
video_dialogue_model="../../../video_dialogue_model"

fairseq-train \
  --save-dir $MODEL_DIR \
  --user-dir $video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --arch baseline-diy-img-transformer \
  --encoder-layers $LAYER \
  --decoder-layers $LAYER \
  --encoder-embed-dim 512 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 8000 \
  --batch-size 128 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $LR \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 60 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --use-img \
  --num-workers 0 \
  #--input-img-dim 1000
exit
# generate system predictions to OUTPUT
MODEL_PATH="${MODEL_DIR}/checkpoint_best.pt"
OUTPUT="${MODEL_DIR}/gen.out"
python /OpenViDial/repo/train/generate.py \
  --user-dir $video_dialogue_model \
  --num-workers 0 \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --path $MODEL_PATH \
  --beam 5 \
  --batch-size 32 \
  --remove-bpe \
  --gen-subset "test" >$OUTPUT 2>&1 & tail -f $OUTPUT 2>&1 \
  
