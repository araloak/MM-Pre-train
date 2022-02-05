# hyper-params
LR=1e-2
DROPOUT=0
LAYER=0
WARMUP=6000
source activate python36

# directory to save models
MODEL_DIR="C:\Users\araloak\Desktop\OpenViDial\models"
# data directory
DATA_DIR="/c/Users/araloak/Desktop/OpenViDial/pretrain_data/cnn_data/2048dim"
BART_PATH="D:\LM\bart_large_pt\model.pt"
video_dialogue_model="C:\Users\araloak\Desktop\OpenViDial\video_dialogue_model"
TYPE="features"
#  --restore-file $BART_PATH \

fairseq-train \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir $MODEL_DIR \
  --user-dir $video_dialogue_model \
  --task img-cls \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --arch baseline-img-cls-transformer \
  --encoder-layers $LAYER \
  --decoder-layers $LAYER \
  --encoder-embed-dim 1024 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --encoder-learned-pos \
  --decoder-learned-pos \
  --share-all-embeddings \
  --layernorm-embedding \
  --optimizer adam \
  --max-tokens 100 \
  --batch-size 32 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $LR \
  --img-dim 2048 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-04 --warmup-updates $WARMUP \
  --max-epoch 100 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --encoder-ffn-embed-dim 4096 \
  --log-format json \
  --num-workers 0 \
  --use-img \
  --pre-train \
  --sentence-avg \
  --criterion cls_cross_entropy \
  #--save-interval 10 \

#  --sentence-avg \


# generate system predictions to OUTPUT
MODEL_PATH="${MODEL_DIR}/checkpoint_best.th"
#python ./eval.py
