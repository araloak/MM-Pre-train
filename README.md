# Emotion-aware Multimodal Pre-training for Image-grounded Emotional Response Generation



This work is developed upon these projects

- [ShannonAI/OpenViDial: Code, Models and Datasets for OpenViDial Dataset (github.com)](https://github.com/ShannonAI/OpenViDial)
- [EasonCai-Dev/torch_backbones: Unofficial implementations of some classical CNN backbones with pytorch (github.com)](https://github.com/EasonCai-Dev/torch_backbones)
- [mabdullah1994/Text-Classification-with-BERT-PyTorch: A text classifier fine tuned on pre-trained BERT for Sarcasm Detection in News Headlines (PyTorch Implementation) (github.com)](https://github.com/mabdullah1994/Text-Classification-with-BERT-PyTorch)



The implementation is based on fairseq framework with pytorch. We delete the identification-related information in all the scripts.



./data directory contains scripts describing the formation of datasets.

./extract_features directory contains scripts regarding the collecting and pre-processing of large image datasets.

./model directory contains scripts defining the model structures been studied.

./resnet directory contains the scripts that model, train and predict of our image encoder

./sentiment_predictor directory contains the implementation of Bert based sentiment predictor for the **C-I2T** task.

./tasks directory contains scripts defining four pre-training tasks and the downstream task. 

./scripts directory contains shell scripts related to the pre-training and finetuning of different tasks.

