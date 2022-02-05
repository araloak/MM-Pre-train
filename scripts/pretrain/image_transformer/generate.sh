# hyper-params
LR=1e-2
DROPOUT=0
LAYER=3
WARMUP=6000
export CUDA_VISIBLE_DEVICES=0
# directory to save models
MODEL_DIR="../../../models"
DATA_DIR=""
TYPE="features"
video_dialogue_model="../../video_dialogue_model"


# generate system predictions to OUTPUT
MODEL_PATH="${MODEL_DIR}/checkpoint_best.pt"
OUTPUT="${MODEL_DIR}/gen.out"
python /c/Users/araloak/Desktop/OpenViDial/repo/train/generate.py \
  --user-dir $video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --path $MODEL_PATH \
  --beam 5 \
  --batch-size 32 \
  --remove-bpe \
  --gen-subset "test" >$OUTPUT 2>&1 & tail -f $OUTPUT 2>&1

