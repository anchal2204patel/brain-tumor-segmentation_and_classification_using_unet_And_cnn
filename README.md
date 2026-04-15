# brain-tumor-segmentation_and_classification_using_unet_And_cnn

# 🧠 Brain Tumor Detection using U-Net + CNN

## 📌 Overview

This project implements a **two-stage deep learning pipeline** for brain tumor analysis using MRI images:

1. **Segmentation (U-Net)** → Detects tumor region
2. **Classification (CNN)** → Classifies tumor type

The pipeline combines both models to improve interpretability and performance.

---

## 🚀 Features

* 🔍 Tumor **segmentation** using U-Net
* 🧠 Tumor **classification** using CNN
* 🔗 End-to-end **pipeline integration**
* 🎲 Random image testing
* 📊 Pipeline accuracy evaluation
* 📈 Confusion matrix support

---

## 🗂️ Dataset Structure

```
DATASET/
│
├── Segmentation/
│   ├── Glioma/
│   ├── Meningioma/
│   └── Pituitary tumor/
│
└── classification/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── notumor/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── notumor/
```

---

## ⚙️ Models Used

### 🧩 U-Net (Segmentation)

* Input: `256 × 256 × 1`
* Output: Binary tumor mask
* Loss: Binary Crossentropy
* Activation: Sigmoid

### 🧠 CNN (Classification)

* Input: `128 × 128 × 1`
* Output: 4 classes

  * glioma
  * meningioma
  * pituitary
  * notumor
* Loss: Categorical Crossentropy

---

## 🔄 Pipeline Workflow

```
Input Image
     ↓
Preprocessing (Grayscale + Resize)
     ↓
U-Net → Tumor Mask
     ↓
(Visualization Only)
     ↓
Original Image → CNN
     ↓
Final Prediction
```

⚠️ Note:
CNN is trained on **original images**, so classification is performed on the original image (not masked output).

---

## 🧪 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Pipeline on Random Image

```python
test_random_image()
```

---

### 4️⃣ Evaluate Pipeline Accuracy

```python
evaluate_pipeline()
```

---

## 📊 Sample Output

```
Image: glioma/Tr_001.jpg
Actual: glioma
Prediction: glioma (0.91)
Correct ✅
```

---

## 📈 Results

* ✅ CNN Accuracy: ~90%
* ✅ U-Net Accuracy: ~99%
* ✅ Pipeline Accuracy: ~89%

*(Update with your actual results)*

---

## 📸 Visualization

The pipeline displays:

* Input MRI image
* Predicted tumor mask
* Final processed output

---

## 🧠 Key Learnings

* Combining segmentation + classification improves interpretability
* Model mismatch (masked vs original input) affects performance
* Proper preprocessing consistency is critical

---

## 🔮 Future Improvements

* Train CNN on masked images
* Use transfer learning (ResNet, EfficientNet)
* Deploy as web app (Streamlit / Flask)
* Real-time prediction system

---

## 👩‍💻 Author

**Avni**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
