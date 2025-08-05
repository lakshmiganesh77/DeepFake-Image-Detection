# 🧠 DeepFake Image Detection – A Computer Vision Method for Synthetic Media Identification

In 2023, a hyper-realistic deepfake video featuring Indian actress Rashmika Mandanna went viral online, depicting her in a compromising scenario falsely. While completely AI-generated, the video was eerily realistic and prompted across-country discussions on digital ethics, consent, and disinformation. The event, as with most others, demonstrates the increasing abuse of deepfake technology to sway public opinion and destroy reputations.

Deepfakes are artificially created media—images, videos, and audio—produced by deep learning models like Generative Adversarial Networks (GANs). They mimic human expressions, voice, and face movements with breathtaking accuracy, rendering it ever more challenging to separate real from synthetic content. With the technology becoming more accessible, so is its potential for malignancy in areas like fake news, blackmail, election meddling, and tampering with evidence. Accurately developing real-time detection systems, therefore, has become imperative.

---

## 📚 About the Project

Deepfakes are synthetic media where a person's likeness is replaced with someone else’s. This project aims to detect such manipulated images using deep learning. Using a supervised learning approach, we train a CNN to distinguish between real and fake images using labeled image datasets.

The model is trained on a dataset structured into `Train/real`, `Train/fake`, `Test/real`, and `Test/fake` folders, where each subfolder contains corresponding labeled images. It uses standard computer vision techniques like image preprocessing, normalization, and data augmentation to improve performance and generalization.

---

## 🧠 What is a CNN?

A **Convolutional Neural Network (CNN)** is a type of deep learning model used primarily in image classification, object detection, and image recognition tasks. CNNs consist of layers such as:
- **Convolutional Layers** to extract features
- **Pooling Layers** to reduce dimensionality
- **Fully Connected Layers** for classification

CNNs automatically learn spatial hierarchies of features from input images, making them perfect for tasks like deepfake detection.

---

## 🛠️ Technologies & Libraries Used

- **TensorFlow** & **Keras** – For building and training the CNN  
- **Matplotlib** – For visualization of training results and predictions  
- **Scikit-learn** – For model evaluation (confusion matrix, classification report)  
- **NumPy** – For array and numerical operations  
- **Google Colab** – Cloud-based platform for model development  
- **ImageDataGenerator** – For augmenting image data  

---

## 📁 Dataset Structure

The project expects the dataset to be organized as follows:

```
deepfake/
├── Train/
│ ├── real/
│ └── fake/
└── Test/
├── real/
└── fake/
```  

---

## 🔍 Project Workflow & Code Block Descriptions

| 🔢 Step | Description |
|--------|-------------|
| 1️⃣ | **Install dependencies** – TensorFlow, matplotlib, scikit-learn |
| 2️⃣ | **Import libraries** – Import necessary Python packages |
| 3️⃣ | **Mount Google Drive** – Access dataset stored in Google Drive |
| 4️⃣ | **Set parameters and paths** – Define batch size, image size, and directory paths |
| 5️⃣ | **Load data** – Read images from folder structure, resize, normalize |
| 6️⃣ | **Train-validation split** – Create a 90-10 split for model validation |
| 7️⃣ | **Data augmentation** – Apply rotation, zoom, shift, and flip to training images |
| 8️⃣ | **Build CNN model** – Define a 3-layer convolutional neural network with dropout |
| 9️⃣ | **Compile and train** – Use Adam optimizer and cross-entropy loss |
| 🔟 | **Plot accuracy/loss** – Visualize training vs validation performance |
| 1️⃣1️⃣ | **Evaluate on test set** – Predict on unseen data and compute accuracy |
| 1️⃣2️⃣ | **Confusion matrix & report** – Print performance metrics |
| 1️⃣3️⃣ | **Single image prediction** – Upload and classify any image in real time |

---

---

## 📊 Model Evaluation

The trained model is evaluated using:
- **Test accuracy**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1 Score)
- **Loss and Accuracy Curves**

---

## 🖼️ Real-Time Image Prediction

The final part allows the user to upload an image and classify it as real or fake. The model returns:
- Predicted label (`real` or `fake`)
- Confidence score (in %)
- The image with prediction title

---

## 📌 Key Highlights

- End-to-end pipeline for image classification  
- Data loading, augmentation, and model training included  
- Interactive prediction on user-uploaded image  
- Easily extendable to video deepfake detection  

---

## 🚀 How to Use

1. Upload the dataset to Google Drive in the correct folder structure.  
2. Open the notebook in Google Colab.  
3. Run all cells step-by-step.  
4. Upload a test image to classify in real-time.  

---

## 📎 License

This project is free and open-source for educational and research use.

---

## 🙌 Acknowledgements

This project was developed as part of my learning journey into **Computer Vision** and **Deep Learning**. Inspired by real-world concerns of digital authenticity, it demonstrates the potential of CNNs and TensorFlow in practical AI applications.

