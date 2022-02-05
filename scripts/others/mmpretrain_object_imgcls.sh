# hyper-params
LR=1e-1
DROPOUT=0.3
LAYER=3
WARMUP=6000
source activate python36
# directory to save models
MODEL_DIR="C:\Users\araloak\Desktop\OpenViDial\models"
# data directory
DATA_DIR="/c/Users/araloak/Desktop/OpenViDial/pretrain_data/obj_data"
BART_PATH="D:\LM\bart_large_pt\model.pt"
USER_DIR="C:\Users\araloak\Desktop\OpenViDial\video_dialogue_model"
TYPE="objects"
fairseq-train \
  --restore-file $BART_PATH \
  --num-workers 0 \
  --pre-train \
  --reset-optimizer --reset-dataloader --reset-meters \
  --task "img-cls" \
  --arch baseline-obj-cls-transformer \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --save-dir $MODEL_DIR \
  --user-dir $USER_DIR \
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
  --batch-size 3 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $LR \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-05 --warmup-updates $WARMUP \
  --max-epoch 200 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --encoder-ffn-embed-dim 4096 \
  --log-format json \
