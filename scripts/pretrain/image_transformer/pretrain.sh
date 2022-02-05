# hyper-params
LR=1e-2
DROPOUT=0.3
LAYER=3
WARMUP=6000
source activate python36

# directory to save models
MODEL_DIR="../../../models/2/"
# data directory
DATA_DIR=""
#BART_PATH="D:\LM\bart_large_pt\model.pt"
video_dialogue_model="../../video_dialogue_model"
TYPE="features"
#  --restore-file $BART_PATH \

fairseq-train \
  --save-dir $MODEL_DIR \
  --user-dir $video_dialogue_model \
  --task img-cls \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --arch baseline-img-cls-transformer \
  --encoder-layers $LAYER \
  --decoder-layers $LAYER \
  --encoder-embed-dim 512 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 8000 \
  --batch-size 32 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $LR \
  --img-dim 1000 \
  --pretrain-img-dim 1000 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-04 --warmup-updates $WARMUP \
  --max-epoch 40 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --log-format json \
  --num-workers 0 \
  --sentence-avg \
  --criterion cls_cross_entropy \
  #--save-interval 10 \
#python ./eval.py
