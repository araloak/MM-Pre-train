export PATH=$PATH:$TF
TASK=/c/Users/araloak/Desktop/OpenViDial/origin_data

for SPLIT in test train valid test
do
  python -m multiprocessing_bpe_encoder \
  --encoder-json encoder.json \
  --vocab-bpe vocab.bpe \
  --inputs "$TASK/$SPLIT.src.txt" \
  --outputs "$TASK/$SPLIT.src.txt.bpe" \
  --workers 1 \
  --keep-empty;
done