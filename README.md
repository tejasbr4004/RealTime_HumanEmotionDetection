# 🎭 RealTime Human Emotion Detection

A deep learning project that detects **7 human emotions in real time** using a webcam. Built with a Convolutional Neural Network (CNN) trained on facial expression images, and deployed using OpenCV for live video inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Emotions Detected](#emotions-detected)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Training the Model](#training-the-model)
  - [Running Real-Time Detection](#running-real-time-detection)
- [File Descriptions](#file-descriptions)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## 🧠 Overview

This project uses a deep **Convolutional Neural Network (CNN)** to classify human facial expressions into one of 7 emotion categories from a live webcam feed. The pipeline works in two stages:

1. **Training** — The CNN is trained on labeled facial images (48×48 grayscale) using a Jupyter Notebook.
2. **Inference** — A Python script captures webcam frames, detects faces using OpenCV's Haar Cascade, and predicts the emotion in real time.

---

## 😄 Emotions Detected

| Label | Emotion   |
|-------|-----------|
| 0     | Angry     |
| 1     | Disgust   |
| 2     | Fear      |
| 3     | Happy     |
| 4     | Neutral   |
| 5     | Sad       |
| 6     | Surprise  |

---

## 📁 Project Structure

```
RealTime_HumanEmotionDetection/
│
├── images/
│   ├── train/                  # Training images organized by emotion label
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                   # Test images organized by emotion label
│       └── (same structure as train/)
│
├── Human_Emotion_Detection.ipynb   # Jupyter Notebook for training the model
├── realtimedetection.py            # Real-time webcam emotion detection script
├── emotiondetector.json            # Saved model architecture
├── emotiondetector.h5              # Saved model weights
├── requirements.txt                # Python dependencies
└── README.md
```

> ⚠️ **Note:** The `images/` folder and `emotiondetector.h5` file are **not included** in this repository due to size constraints. See [Dataset Setup](#dataset-setup) and [Training the Model](#training-the-model) to generate them.

---

## ⚙️ How It Works

```
Webcam Feed
    │
    ▼
Convert to Grayscale
    │
    ▼
Haar Cascade Face Detection  ──► Draw Bounding Box on Frame
    │
    ▼
Crop & Resize Face to 48×48 px
    │
    ▼
Normalize Pixel Values (÷ 255)
    │
    ▼
CNN Model Prediction
    │
    ▼
Display Emotion Label on Screen
```

---

## 🏗️ Model Architecture

The CNN is built using Keras/TensorFlow and consists of the following layers:

| Layer Type        | Filters / Units | Kernel / Pool | Activation | Dropout |
|-------------------|-----------------|---------------|------------|---------|
| Conv2D            | 128             | 3×3           | ReLU       | 0.4     |
| MaxPooling2D      | —               | 2×2           | —          | —       |
| Conv2D            | 256             | 3×3           | ReLU       | 0.4     |
| MaxPooling2D      | —               | 2×2           | —          | —       |
| Conv2D            | 512             | 3×3           | ReLU       | 0.4     |
| MaxPooling2D      | —               | 2×2           | —          | —       |
| Conv2D            | 512             | 3×3           | ReLU       | 0.4     |
| MaxPooling2D      | —               | 2×2           | —          | —       |
| Flatten           | —               | —             | —          | —       |
| Dense             | 512             | —             | ReLU       | 0.4     |
| Dense             | 256             | —             | ReLU       | 0.3     |
| Dense (Output)    | 7               | —             | Softmax    | —       |

**Training configuration:**
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
- Batch Size: `128`
- Epochs: `100`
- Input Shape: `(48, 48, 1)` — grayscale images

---

## 🚀 Getting Started

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.8 or above — [Download Python](https://www.python.org/downloads/)
- pip (comes with Python)
- A working webcam (for real-time detection)
- Git — [Download Git](https://git-scm.com/)

---

### Installation

**Step 1 — Clone the repository:**

```bash
git clone https://github.com/tejasbr4004/RealTime_HumanEmotionDetection.git
cd RealTime_HumanEmotionDetection
```

**Step 2 — (Recommended) Create a virtual environment:**

```bash
# Create virtual environment
python -m venv venv

# Activate it — on Windows:
venv\Scripts\activate

# Activate it — on macOS/Linux:
source venv/bin/activate
```

**Step 3 — Install dependencies:**

```bash
pip install -r requirements.txt
```

---

### Dataset Setup

This project uses facial expression images in the following format:

```
images/
├── train/
│   ├── angry/      ← place angry training images here
│   ├── happy/      ← place happy training images here
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

You can use the **FER-2013 dataset**, which is publicly available on Kaggle:

1. Go to: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download and extract it
3. Organize the images into the `images/train/` and `images/test/` folders as shown above

> Each sub-folder name becomes the emotion label automatically (e.g., images inside `happy/` are labelled as "happy").

---

### Training the Model

Once the dataset is ready, open the Jupyter Notebook to train the model:

```bash
jupyter notebook Human_Emotion_Detection.ipynb
```

Run all the cells from top to bottom. At the end, two files will be saved in your project folder:

- `emotiondetector.json` — the model architecture
- `emotiondetector.h5` — the trained model weights

> ⏱️ Training for 100 epochs may take a while depending on your hardware. Using a GPU is recommended for faster training.

---

### Running Real-Time Detection

Once the model files (`emotiondetector.json` and `emotiondetector.h5`) exist, run:

```bash
python realtimedetection.py
```

A window will open showing your webcam feed. Detected faces will be highlighted with a blue bounding box, and the predicted emotion label will be displayed above the face in red text.

**To quit:** Press `Esc` on your keyboard.

---

## 📄 File Descriptions

| File | Description |
|------|-------------|
| `Human_Emotion_Detection.ipynb` | Full training pipeline — data loading, preprocessing, model building, training, and evaluation |
| `realtimedetection.py` | Loads the trained model and runs live emotion detection via webcam |
| `emotiondetector.json` | JSON file containing the serialized model architecture |
| `emotiondetector.h5` | HDF5 file containing the trained model weights |
| `requirements.txt` | All Python package dependencies needed to run this project |

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| Python | Core programming language |
| TensorFlow / Keras | Building and training the CNN model |
| OpenCV (`cv2`) | Webcam capture, face detection, image display |
| NumPy | Array operations and image preprocessing |
| Pandas | Loading and managing image paths and labels |
| Scikit-learn | Label encoding for emotion classes |
| tqdm | Progress bar during feature extraction |
| Matplotlib | Visualizing prediction results in the notebook |
| Jupyter Notebook | Interactive training environment |

---

## 🐛 Troubleshooting

**Webcam not opening?**
- Make sure no other application is using your webcam.
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `realtimedetection.py` if you have multiple cameras.

**`emotiondetector.h5` not found?**
- You need to train the model first by running the Jupyter Notebook completely.

**`ModuleNotFoundError`?**
- Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**Low accuracy or wrong predictions?**
- The model was trained on 48×48 grayscale face crops. Ensure your lighting is adequate and you are clearly visible to the camera.
- Training for more epochs or using data augmentation can improve accuracy.

**Keras version conflicts?**
- This project uses `keras` with `tensorflow` as the backend. Avoid mixing standalone `keras` with `tf.keras` in the same environment.

---

## 🔮 Future Improvements

- [ ] Add a confidence percentage display alongside the emotion label
- [ ] Support detection of multiple faces simultaneously
- [ ] Build a web-based interface using Flask or Streamlit
- [ ] Improve accuracy with data augmentation and transfer learning
- [ ] Export the model to ONNX or TensorFlow Lite for edge deployment
- [ ] Add a logging system to record emotions over time

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — the facial expression dataset used for training
- [OpenCV](https://opencv.org/) — for real-time computer vision
- [TensorFlow / Keras](https://www.tensorflow.org/) — for the deep learning framework
