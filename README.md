#kansai_train

This project is a high-accuracy railway vehicle classification system developed on NVIDIA Jetson Orin Nano, designed to automatically identify 31 different train series running in the Kansai region (JR West & Osaka Metro).

##Project Overview
- **AI Architecture**: PyTorch (`torchvision.models.resnet18`)
- **Training**: 100 Epochs trained on GPU (Orin Nano) to recognize distinct facial components, window frames, and stripe patterns of each vehicle.
- **Input Preprocessing**: Automatic centered 1:1 face-cropping is highly recommended to eliminate background noise (such as station platforms or overhead wires).

##Requirements
To run this project, you need the following environment and libraries installed inside your Jetson environment:
- NVIDIA Jetson Orin Nano
- `dustynv/jetson-inference` Docker container
- Python 3.10+
- `torch` & `torchvision`
- `opencv-python` (`cv2`)
- `Pillow` (`PIL`)

---

##Target Classes (31 Train Series)
The system can perfectly distinguish the following 31 classes:

### Osaka Metro (11 Series)
`metro21` / `metro22` / `metro23` / `metro25` / `metro30000A` / `metro31000` / `metro32000` / `metro400` / `metro66` / `metro70` / `metro80`

### JR West & Express (20 Series)
`series113` / `series117` / `series189` / `series207` / `series221` / `series223` / `series225` / `series227` / `series271` / `series281` / `series283` / `series285` / `series287` / `series289` / `series321` / `series323` / `series681` / `series683` / `series87` / `seriesHOT7000`

---

## How to Use

### Step 1: Launch the Docker Container with synced data volume
Run this command from your Orin host terminal to start the environment:
```bash
cd ~/jetson-inference && ./docker/run.sh --volume /home/nvidia/data:/opt/jetson-inference/python/training/classification/data
```

### Step 2: Move to the correct workspace directory
Inside the container, move to the classification directory:
```bash
cd /opt/jetson-inference/python/training/classification/
```

### Step 3: Prepare your target picture
1. Crop your target train photo into a **1:1 square (focused on the train's face)**.
2. Rename the picture file to **`test_train.jpg`**.
3. Place/overwrite this file under the **`data/jr_west/`** directory.

### Step 4: Run the identification program
Execute the prediction script:
```bash
python3 data/jr_west/jr_predict_31classes.py
```

###Output Results
- The AI will immediately output the classified train series name and its probability on your terminal.
- It will automatically save a processed result image at **`data/jr_west/result_train.jpg`**, labeled with a clean, rescaled green text (size 0.8) showing the correct prediction.

---

##  Demo Video
I deployed and tested this system on a video inference task. You can watch the full demonstration here:
[(https://drive.google.com/file/d/1u6z-nxq-Q88zivFK6C9FqIgUKwNxJE4g/view?usp=drive_link)]
