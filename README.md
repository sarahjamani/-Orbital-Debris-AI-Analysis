
# Orbital Debris AI Analysis 🛰️
AI-based analysis of NASA's orbital debris dataset — identifying growth trends, classifying debris by reusability, and predicting future debris population through 2050 using machine learning.

## 📌 Overview

Space debris is accumulating rapidly, threatening both active satellites and future space missions. This project applies machine learning to NASA's orbital debris data to:

- Identify the fastest-growing debris categories
- Classify debris as hazardous, partially recyclable, or reusable
- Predict total orbital debris growth through 2050
- Estimate how much existing debris could realistically be recovered or reused

## 🎯 Objectives

1. Clean and pre-process NASA's orbital debris dataset
2. Identify the fastest-growing debris categories
3. Train a Decision Tree classifier to categorize debris (hazardous / partially recyclable / reusable)
4. Train a Linear Regression model to forecast debris growth through 2050
5. Estimate the reusability potential of major debris types (D11, D7, D8)
6. Gather public opinion via survey on AI-driven space sustainability solutions

## 🧠 Methods

| Step | Approach |
|---|---|
| Data Cleaning | Python (pandas) |
| Classification | Decision Tree (80/20 train-test split) |
| Prediction | Linear Regression (Year vs. Total Debris) |
| Reusability Estimation | Weighted heuristic: 40% (D11) + 30% (D7) + 20% (D8) |

## 📊 Key Results

- **Decision Tree accuracy:** 100% (652 training samples, 164 testing samples)
- **Reusable debris estimate:** ~32.3% of total debris is potentially reusable
- **Growth trend:** Sharp increase in debris volume after year 2000
- **Survey finding:** 100% of respondents believed AI can help address space sustainability challenges

## 🛠️ Tech Stack

- Python
- pandas, scikit-learn
- Matplotlib / visualization libraries

## 📁 Repository Structure

```
├── data/              # NASA orbital debris dataset
├── notebooks/scripts/ # Data cleaning, classification, and prediction code
├── README.md
└── LICENSE
```

## 📄 Full Research Paper

The full write-up — including literature review, methodology, and detailed results — is published as an open-access preprint on Zenodo:
🔗 *[https://zenodo.org/records/22016828]*

## 👩‍💻 Author

**Sarah Salim Jamani**
Department of Computer Science, SZABIST University, Karachi, Pakistan
📧 salimjamani9@gmail.com

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🔭 Future Work

- Deep learning models (LSTM, GRU) for improved debris growth prediction
- Computer vision for detecting small/untracked debris via satellite imagery
- Real-time AI-based collision avoidance systems
