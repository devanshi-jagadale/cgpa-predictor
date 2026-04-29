# 🎓 CGPA Predictor

### Machine Learning + OCR-powered Academic Forecasting System

Predict a student’s final CGPA **early in their academic journey** using machine learning and automated grade extraction.

---

## 🚀 Overview

Students usually realize their academic standing **too late**. This project solves that by predicting final CGPA using:

* 📄 OCR-based grade sheet parsing
* 🧠 Machine learning (Random Forest)
* 🔁 Two-stage prediction pipeline

The system works with both:

* 📂 Uploaded PDFs/images (automatic extraction)
* ✍️ Manual grade input

---

## 🎯 Problem Statement

* ❌ No early warning system for academic performance
* ❌ Manual grade analysis is slow and error-prone
* ❌ No reliable projection of future CGPA

---

## 💡 Solution

We built a **two-stage ML pipeline** that:

* Extracts SGPA and grade data
* Engineers meaningful features
* Predicts future SGPA and final CGPA

> Result: Students and advisors get **early insights** into academic outcomes

---

## 🧠 Architecture

### 🔹 Stage 1 — Early Prediction

* **Input:** S1–S4 SGPA + grade stats
* **Output:** Predicted S5 & S6
* **Model:** RandomForestRegressor
* **Dataset:** `batch_22.csv`

---

### 🔹 Stage 2 — Final Prediction

* **Input:** S1–S6 SGPA + grade stats
* **Output:** Predicted S7, S8 & CGPA
* **Model:** RandomForestRegressor
* **Dataset:** `1.csv`

---

## 📊 Model Performance

* **R² Score:**

  * Stage 1 → 0.624
  * Stage 2 → 0.804

* **MAE:**

  * Stage 1 → 0.755
  * Stage 2 → 0.492

---

## 🔍 OCR Pipeline

1. 📂 Upload PDF/Image
2. 🔎 Extract text using OCR
3. 🧩 Parse structured data
4. ⚙️ Generate features
5. 📈 Predict CGPA

Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg` 

---

## ⚙️ Features Engineered

* `sgpa_trend` → performance slope
* `avg_4` → average of first 4 semesters
* `max_4`, `min_4` → best/worst SGPA
* `num_FF` → failed subjects
* `num_AA` → top grades

---

## 🖥️ Tech Stack

* **Backend:** FastAPI
* **ML:** scikit-learn (Random Forest)
* **OCR:** pytesseract
* **Frontend:** HTML, CSS, JavaScript
* **Deployment:** Uvicorn

---

## 📁 Project Structure

```
ML_PROJECT/
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── train.py
│   ├── utils/
│   │   ├── features.py
│   │   ├── ocr.py
│   │   └── parser.py
│
├── data/
│   └── sample.csv
│
├── frontend/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Project

### 1️⃣ Clone the repo

```
git clone https://github.com/your-username/cgpa-predictor.git
cd cgpa-predictor
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run backend

```
uvicorn backend.app:app --reload
```

### 4️⃣ Open frontend

Open `frontend/index.html` in your browser

---

## 📡 API Endpoints

### 🔹 `/predict`

* Input: PDF/Image
* Output: `predicted_cgpa`, extracted data

---

### 🔹 `/predict_manual`

* Input: JSON (SGPA + grades)
* Output: `predicted_cgpa`, extracted data

---

## 📦 Model Files

Model files are not included due to size.

👉 Add your download link here

Place them in:

```
backend/model/
```

---

## 🌐 System Flow

User → Frontend → FastAPI → OCR → Feature Engine → Model → Prediction

---

## ✨ Highlights

* 📄 Zero manual entry using OCR
* 🔁 Two-stage ML pipeline
* ⚡ Real-time predictions via API
* 🎨 Clean frontend with live visualization

---

## 🚧 Future Improvements

* Model accuracy improvement
* Better OCR robustness
* Full web deployment
* Authentication system

---

## 👨‍💻 Authors

* Devanshi Jagadale
* Rahul Moolchandani


---
