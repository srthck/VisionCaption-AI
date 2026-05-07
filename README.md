---
title: VisionCaption AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---<div align="center">
  <h1>👁️ Visionary AI</h1>
  <p><strong>Production-Grade Image Captioning System (CNN-LSTM)</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📌 Project Overview
**Visionary AI** is an advanced, production-ready machine learning application that automatically generates descriptive captions for uploaded images and synthesizes them into speech. 

Built using a hybrid **CNN-LSTM** architecture, the system leverages a pre-trained **VGG16** model for deep image feature extraction, coupled with an **LSTM** sequence model for natural language generation. The application is served via a robust, highly-optimized **Streamlit** interface designed for deployment in scalable environments.

This repository demonstrates senior-level software engineering practices, including modular architecture, clean code principles, robust error handling, and deployment readiness.

---

## 🚀 Key Features
- **Deep Feature Extraction**: Utilizes transfer learning via VGG16 to map images into a dense 4096-dimensional feature space.
- **Sequence Generation**: Employs an LSTM-based recurrent neural network to generate contextually accurate textual descriptions token-by-token.
- **Audio Synthesis**: Integrates Google Text-to-Speech (gTTS) to provide accessible audio playback of the generated captions.
- **Optimized Caching**: Implements aggressive resource caching (`@st.cache_resource`) to load heavy neural networks only once, ensuring near-instantaneous inference across sessions.
- **Production-Grade UI**: Features a modern, responsive, and cleanly typed user interface with graceful degradation and comprehensive exception handling.

---

## 🏗️ Architecture

The codebase has been strictly modularized to separate the Deep Learning inference logic from the Presentation (UI) layer:

```text
project/
├── app.py                      # Streamlit UI and Presentation Layer
├── requirements.txt            # Minimal Production Dependencies
├── .gitignore                  # Standard Python/ML Gitignore
├── artifacts/                  # Trained ML Models (Git LFS)
│   ├── model.h5                # VGG16+LSTM weights
│   └── tokenizer.pkl           # Keras text tokenizer
├── models/                     # Deep Learning Models Module
│   ├── feature_extractor.py    # VGG16 Loading and Inference
│   └── caption_model.py        # LSTM and Tokenizer Management
├── utils/                      # Helper Utilities Module
│   ├── image_processing.py     # Tensor Preprocessing for VGG16
│   ├── caption_generator.py    # Token sequence generation and cleaning
│   └── audio_generator.py      # TTS Audio Synthesis
└── assets/                     # Auto-generated Audio Outputs
```

### Engineering Decisions:
1. **Separation of Concerns**: Machine learning logic is decoupled from Streamlit, allowing models to be easily ported to a FastAPI or Flask backend in the future.
2. **Safe Resource Management**: Tokenizers and `.h5` model files are loaded with strict `try-except` blocks to prevent catastrophic runtime failures.
3. **Optimized I/O**: Uploaded images are kept in memory and safely reset during tensor conversion to prevent redundant disk writing.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/visionary-ai.git
cd visionary-ai
```

### 2. Set up a virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Provide the trained models
Ensure the following files are placed in the root directory (these are ignored via `.gitignore` due to size):
- `model.h5` (Trained LSTM weights)
- `tokenizer.pkl` (Fitted Keras Tokenizer)

### 5. Run the application
```bash
streamlit run app.py
```

---

## 🌐 Deployment
This project is fully configured for deployment on **Streamlit Community Cloud** or **Hugging Face Spaces**. 
1. Push the code to GitHub.
2. Connect the repository to your chosen PaaS.
3. *Note*: You may need to use Git LFS (Large File Storage) to upload `model.h5` and `tokenizer.pkl` to your repository.

---

## 🔮 Future Improvements
While currently production-stable, future iterations of this project could include:
- **Transformer Migration**: Upgrading the LSTM sequence generator to a Vision-Transformer (ViT) or BLIP-based architecture for state-of-the-art accuracy.
- **REST API Extraction**: Decoupling the Streamlit frontend from the ML backend by wrapping the `models/` directory in a FastAPI service.
- **Dockerization**: Adding a `Dockerfile` for standardized, containerized deployments across AWS/GCP.

---

<div align="center">
  <i>Engineered for Scalability and Readability.</i>
</div>