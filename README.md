# Handwritten Alphabet Predictor (A–Z)

A browser-based AI application that recognizes handwritten alphabet letters drawn by the user.

The system uses a Convolutional Neural Network (CNN) trained on the **EMNIST dataset** and runs directly in the browser using **TensorFlow.js**.
Users can draw a letter on a canvas and the model predicts the corresponding alphabet character.

The model is **case-insensitive**, meaning that whether the user draws **uppercase or lowercase**, the system predicts the correct letter (A–Z).

---

# Live Demo

Deployed application: https://alphabet-ai.vercel.app

---

# Features

• Draw letters directly in the browser
• Real-time AI prediction
• Case-insensitive recognition (A–Z)
• Runs completely in the browser (no backend server)
• Lightweight TensorFlow.js model

---

# Tech Stack

### Machine Learning

* TensorFlow
* TensorFlow Datasets
* EMNIST (Extended MNIST)

### Frontend

* Next.js
* React
* HTML Canvas API

### Deployment

* TensorFlow.js
* Vercel

---

# Model Training

The CNN model was trained using the **EMNIST ByClass dataset**.

Original dataset classes:

```
0–9   : digits
10–35 : uppercase letters
36–61 : lowercase letters
```

For this project, uppercase and lowercase were **merged into 26 classes** so that:

```
A / a → A
B / b → B
...
Z / z → Z
```

This makes the prediction independent of case.

---

# Model Architecture

Convolutional Neural Network:

```
Input (28x28 grayscale)

Conv2D (32)
BatchNorm
ReLU
MaxPooling

Conv2D (64)
BatchNorm
ReLU
MaxPooling

Conv2D (128)
BatchNorm
ReLU
MaxPooling

Flatten
Dense (256)
Dropout
Dense (26 softmax)
```

Test Accuracy:

**~94.9%**

---

# Project Structure

```
alphabet-ai
│
├── app/                     # Next.js frontend
│
├── public/
│   └── model/               # TensorFlow.js model
│       ├── model.json
│       └── group1-shard1of1.bin
│
├── training/                # Model training scripts
│   ├── train_26.py
│   └── requirements.txt
│
├── README.md
└── package.json
```

---

# Running Locally

Clone the repository:

```
git clone https://github.com/Oghuz20/alphabet-ai.git
cd alphabet-ai
```

Install dependencies:

```
npm install
```

Run development server:

```
npm run dev
```

Open in browser:

```
http://localhost:3000
```

---

# How Prediction Works

1. User draws a letter on the canvas.
2. The canvas image is resized to **28x28 pixels**.
3. The image is converted to **grayscale**.
4. Pixel values are normalized.
5. The TensorFlow.js model predicts probabilities for **26 letters**.
6. The letter with the highest probability is displayed.

---

# Future Improvements

• Improve real drawing accuracy using **centering and scaling preprocessing**
• Add **top-3 predictions display**
• Improve UI/UX design
• Add **mobile drawing support**
• Optimize model size for faster loading

---

# Author

**Oghuz Hasanli**

Machine Learning & AI student passionate about building real-world AI systems.
