# hyper-params
PRETRAIN_LR=3e-4
DROPOUT=0.3
PRETRAIN_LAYER=3
WARMUP=6000
export CUDA_VISIBLE_DEVICES=3
echo $PRETRAIN_LR
echo $DROUPOUT
#directory to save models
PRETRAIN_MODEL_DIR=""
#FINETUNE_MODEL_DIR=""

PRETRAIN_DATA_DIR=""
FINETUNE_DATA_DIR=""
PRETRAIN_MODEL_NAME=$PRETRAIN_MODEL_DIR"/checkpoint_best.pt"

#BART_PATH=""
video_dialogue_model=""
TYPE="features"
echo ""
fairseq-train \
  --save-dir $PRETRAIN_MODEL_DIR \
  --user-dir $video_dialogue_model \
  --task img-cls \
  --img-type $TYPE \
  --data-dir $PRETRAIN_DATA_DIR \
  --arch baseline-img-cls-transformer \
  --encoder-layers $PRETRAIN_LAYER \
  --decoder-layers $PRETRAIN_LAYER \
  --encoder-embed-dim 512 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 8000 \
  --batch-size 32 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $PRETRAIN_LR \
  --img-dim 1000 \
  --pretrain-img-dim 1000 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-04 --warmup-updates $WARMUP \
  --max-epoch 10 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --sentence-avg \
  --criterion cls_cross_entropy \
  --num-workers 0

# hyper-params
FINETUNE_LR=3e-4
DROPOUT=0.3
FINETUNE_LAYER=3
WARMUP=6000

FINETUNE_MODEL_DIR=$PRETRAIN_MODEL_DIR"/layer${FINETUNE_LAYER}_lr${FINETUNE_LR}_bsz128_drop${DROPOUT}_warmup${WARMUP}"

fairseq-train \
  --restore-file $PRETRAIN_MODEL_NAME \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir $FINETUNE_MODEL_DIR \
  --user-dir $video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $FINETUNE_DATA_DIR \
  --arch baseline-img-transformer \
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
  --max-epoch 20 \
  --keep-last-epochs 5 \
  --ddp-backend=no_c10d \
  --use-img \
  --input-img-dim 1000
  
# generate system predictions to OUTPUT
MODEL_PATH="${FINETUNE_MODEL_DIR}/checkpoint_best.pt"
OUTPUT="${FINETUNE_MODEL_DIR}/gen.out"
python /openvidial/repo/OpenViDial/train/generate.py \
  --user-dir $video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $FINETUNE_DATA_DIR \
  --path $MODEL_PATH \
  --beam 5 \
  --batch-size 32 \
  --remove-bpe \
  --results-path $FINETUNE_MODEL_DIR \
  #--gen-subset "test" >$OUTPUT 2>&1 & tail -f $OUTPUT 2>&1 
