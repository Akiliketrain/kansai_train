#  JR West & Osaka Metro 31-Classes Train Identification AI

This project is a high-accuracy railway vehicle classification system developed on NVIDIA Jetson Orin Nano, running within a customized jetson-inference Docker environment.

##  Project Specifications
- **Framework**: PyTorch (torchvision.models.resnet18)
- **Epochs**: 100 Epochs trained on GPU (Orin Nano)
- **Target Classes**: 31 Classes (20 JR West & 11 Osaka Metro model series)
- **Input Preprocessing**: Automatic centered 1:1 face-cropping to eliminate background noise

##  Dataset & Map Array Synchronization
The model's classification output utilizes a 1-to-1 strict mapping synchronized with `labels.txt` in perfect alphabetical/numerical sort order, preventing index misalignment.

##  How to Run (Inside Docker Container)
1. Place your target image as `data/jr_west/test_train.jpg`.
2. Run the prediction script:
   ```bash
   python3 jr_predict_31classes.py
   ```
3. The AI will output the best classification result on your terminal and save a localized image labeled with the predicted series name and probability at `data/jr_west/result_train.jpg`.
