
ORIGIN_DIR="/c/Users/araloak/Desktop/OpenViDial/origin_data"
OUTPUT_DIR="/c/Users/araloak/Desktop/OpenViDial/preprocessed_data"


python ./preprocess/preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "train" 

python ./preprocess/preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "valid" 


python ./preprocess/preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "test"
