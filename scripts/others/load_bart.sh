# hyper-params
LR=1e-1
DROPOUT=0.3
LAYER=12
WARMUP=6000
source activate python36
# directory to save models
MODEL_DIR="C:\Users\araloak\Desktop\OpenViDial\models"
# data directory
DATA_DIR="/c/Users/araloak/Desktop/OpenViDial/pretrain_data"
TYPE="features"
BART_PATH="D:\LM\bart_large_pt\model.pt"

fairseq-train \
  --restore-file $BART_PATH \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir $MODEL_DIR \
  --user-dir video_dialogue_model \
  --task img-cls \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --arch baseline-img-transformer \
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
  --img-dim 1000 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 20 \
  --no-epoch-checkpoints \
  --ddp-backend=no_c10d \
  --encoder-ffn-embed-dim 4096 \
  --log-format json \
  --num-workers 0 \
  --use-img \
  --pre-train

:<<!
# generate system predictions to OUTPUT
MODEL_PATH="${MODEL_DIR}/checkpoint_best.th"
OUTPUT="${MODEL_DIR}/gen.out"
python ./train/generate.py \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --path $MODEL_PATH \
  --beam 5 \
  --batch-size 32 \
  --remove-bpe \
  --gen-subset "test" >$OUTPUT 2>&1 & tail -f $OUTPUT 2>&1
!