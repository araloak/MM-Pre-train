TASK=/c/Users/araloak/Desktop/OpenViDial/origin_data
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --testpref "${TASK}/val.bpe" \
  --destdir "${TASK}-base-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;