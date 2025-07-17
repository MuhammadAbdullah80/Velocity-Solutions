# Velovity-Solutions
## AI Internship – Week 1 & Week 2 Tasks

This repository contains all my work from **Week 1** and **Week 2** of the Artificial Intelligence Internship by Velocity Solutions.

## 📌 Overview

### Week 1:
Focused on building fundamental understanding of AI and implementing basic intelligent systems using Python. I practiced working with real datasets, rule-based systems, and classic search algorithms.

### Week 2:
Dived into Natural Language Processing (NLP) and text classification using machine learning. Built an end-to-end SMS Spam Detection system using two types of models.

---

## ✅ Tasks Completed

### Week 1

#### 1. Understanding AI, Machine Learning, and Deep Learning
- Wrote a short summary comparing AI, ML, and DL.
- Highlighted their differences and real-world applications.
- Included this in a markdown cell inside the Colab notebook.

#### 2. Data Exploration and Pandas Practice
- Practiced using the Pandas library in Python for loading and analyzing datasets.
- Explored:
  - **Car Evaluation Dataset**
  - **Tic-Tac-Toe Endgame Dataset**
- Performed `.head()`, `.describe()`, and `.value_counts()` to understand data.
- Applied concepts like feature access, conditional filtering, and basic data cleaning.

#### 3. Rule-Based AI Agents
- Built a rule-based recommendation system for car evaluation.
- Applied the rules to every row of the dataset using `apply()` and returned recommendations like:
  - "Highly Recommended"
  - "Acceptable"
  - "Not Recommended"
- Also built a rule-based agent to check if a Tic-Tac-Toe board is a winning state for X.

#### 4. Search Algorithms (Maze Solvers)
- Implemented **Breadth-First Search (BFS)** and **Depth-First Search (DFS)** to solve mazes.
- BFS returns the shortest path.
- DFS explores all possible paths and displays every dead end.
- Visualized the path on a 2D grid using "S", "G", "|", "_", and "#".

---

### Week 2

#### 1. Dataset Loading and Preprocessing
- Used the **SMS Spam Collection Dataset** from UCI ML Repository.
- Cleaned the text: lowercased, tokenized, removed stopwords, punctuation, and applied stemming & lemmatization.
- Created a `clean_text` column for model training.

#### 2. Feature Extraction
- Transformed the cleaned text into numerical vectors using:
  - **Bag-of-Words (CountVectorizer)**
  - **TF-IDF (TfidfVectorizer)**

#### 3. Model Training and Evaluation
- Trained two classifiers:
  - **Multinomial Naive Bayes** on Bag-of-Words
  - **Logistic Regression** on TF-IDF
- Evaluated using Accuracy, Precision, Recall, and F1-Score.

| Model               | Vectorization | Accuracy | F1 (Spam) |
|---------------------|----------------|----------|------------|
| Naive Bayes         | Bag-of-Words   | 96.95%   | 0.89       |
| Logistic Regression | TF-IDF         | 95.96%   | 0.83       |

#### 4. Real-Time Testing Interface
- Built an interactive tool that allows typing a custom SMS message.
- Both models return predictions (spam/ham) along with confidence scores.

#### 5. Model Comparison
- Constructed a side-by-side metric comparison for both models.
- Demonstrated how model choices affect recall vs. precision.

---

## 🧪 Technologies Used
- Python 3
- Google Colab
- Libraries: `pandas`, `nltk`, `sklearn`

---

## 📁 File Structure
- `Week1.ipynb` → Colab notebook for Week 1 tasks
- `Week2.ipynb` → Colab notebook for Week 2 tasks

