# 🚗 Road Accident Severity Detection

A machine learning and deep learning project that predicts the severity of road accidents using structured tabular data and image data. Multiple models are trained and compared — including classical ML algorithms and a custom CNN.

---

## 📌 Objective

Road accidents remain a leading cause of injury and death worldwide. This project builds a classification system to predict accident severity based on factors like driver profile, road conditions, weather, vehicle type, and cause of accident — enabling data-driven insights for road safety analysis.

---

## 📊 Dataset

- **Tabular data:** `Road.csv` — structured records with 25+ features per accident
- **Image data:** `archive.zip` — images organized into train/test folders by class
- **Target variable:** `Accident_severity`

### Key Features Used

| Category | Features |
|---|---|
| Driver | Age band, sex, driving experience |
| Vehicle | Type of vehicle, defect of vehicle, vehicle movement |
| Road | Road alignment, surface type, surface conditions, lanes/medians |
| Environment | Light conditions, weather conditions, area of occurrence |
| Accident | Type of collision, types of junction, cause of accident |
| Time | Hour, minute (extracted from timestamp) |

> Features like `Educational_level`, `Owner_of_vehicle`, `Service_year_of_vehicle`, and `Number_of_casualties` were dropped during preprocessing.

---

## ⚙️ Steps Involved

1. **Data Loading & Exploration** — shape, info, describe, column inspection
2. **Preprocessing**
   - Drop nulls
   - Extract hour and minute from timestamp
   - Drop irrelevant columns
   - Label encode all categorical features
3. **Correlation Analysis** — heatmap to identify feature relationships
4. **Train/Test Split** — 85/15 split with `random_state=13`
5. **Model Training & Evaluation** — 5 ML models + 1 CNN
6. **Model Comparison** — bar chart of all model accuracies

---

## 🤖 Models & Results

### Classical ML (on tabular data)

| Model | Accuracy |
|---|---|
| Logistic Regression | **86%** |
| Random Forest | **86%** |
| Support Vector Machine | **86%** |
| XGBoost | 85% |
| Decision Tree | 78% |

### Deep Learning (on image data)

| Model | Accuracy |
|---|---|
| Custom CNN (3 conv blocks) | 63% |

> The classical ML models significantly outperformed the CNN on this dataset. The CNN's lower accuracy suggests the image dataset alone carries less predictive signal compared to the structured tabular features.

---

## 🧠 CNN Architecture

```
Conv2D(32) → MaxPooling
Conv2D(64) → MaxPooling
Conv2D(128) → MaxPooling
Flatten
Dense(128, ReLU) → Dropout(0.5)
Dense(num_classes, Softmax)
```

- Input size: **128×128 px**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Notebook | Jupyter / Kaggle / Google Colab |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/road-accident-severity-detection.git
cd road-accident-severity-detection
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
```

### 3. Add the data files
- Place `Road.csv` in the project root
- Place `archive (4).zip` in the project root (the notebook extracts it automatically)

### 4. Run the notebook
```bash
jupyter notebook Road_accident_severity_detection.ipynb
```

---

## 📁 Project Structure

```
road-accident-severity-detection/
├── Road_accident_severity_detection.ipynb
├── Road.csv                  # Tabular accident data (add manually)
├── archive (4).zip           # Image dataset (add manually)
├── dataset/                  # Extracted automatically
│   └── data/
│       ├── train/
│       └── test/
└── README.md
```

---

## 📄 License

MIT — free to use and modify.
