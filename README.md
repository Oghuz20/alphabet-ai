# Handwritten Alphabet Predictor (A–Z)

A lightweight AI web app that recognizes **handwritten English letters**. Users draw a letter on a canvas and the model predicts the corresponding **A–Z** character (**case-insensitive**).

**Live Demo:** https://alphabet-ai-mu.vercel.app/

---

## Features
- **Draw & Predict (A–Z)** directly in the browser
- **Case-insensitive recognition** (handles uppercase and lowercase input)
- **Top-3 predictions** with confidence scores
- **Debug view** to preview the 28×28 model input
- No backend required — runs fully on the client using **TensorFlow.js**

---

## Model Overview
- Trained using the **EMNIST ByClass** dataset (letters)
- Model outputs **52 classes**:
  - `A–Z` (26)
  - `a–z` (26)
- To make the app case-insensitive, prediction merges probabilities:
  - `P(letter) = P(uppercase) + P(lowercase)`
  - Final output is **26 letters (A–Z)**

---

## Preprocessing Pipeline
Canvas drawings differ from EMNIST samples, so the app applies preprocessing before inference:

1. **Center & resize** to 28×28  
2. **Invert** (ink becomes high intensity like MNIST/EMNIST)  
3. **Binarization** to reduce noise  
4. **Center by mass** to align the character  
5. **Orientation fixes** (EMNIST alignment + mirror correction)

A small **debug panel** shows what the model receives as input.

---

## Tech Stack
- **Next.js** (App Router)
- **TypeScript**
- **TensorFlow.js** (`@tensorflow/tfjs`)
- **HTML Canvas** for drawing input

---

## Project Structure
```txt
app/
  page.tsx                # UI, preprocessing, and inference logic

public/
  model52/                # TensorFlow.js model files
    model.json
    group1-shard1of1.bin
Running Locally
1) Install dependencies
npm install
2) Start the development server
npm run dev
3) Open in browser
http://localhost:3000
Deployment (Vercel)

This project is deployed on Vercel.

To deploy your own version:

Push the repository to GitHub

Import the repo into Vercel

Deploy (no environment variables required)

Usage Tips

Draw one letter at a time

Write big and centered

If confidence is low, the app shows ? and suggests redrawing

Credits

Dataset: EMNIST (Extended MNIST)

Inference: TensorFlow.js

Author

Oghuz Hasanli