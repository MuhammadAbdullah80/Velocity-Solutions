# Velocity-Solutions

## AI Internship – Week 1, Week 2 & Week 3 Tasks

This repository contains all my work from **Week 1**, **Week 2**, and **Week 3** of the Artificial Intelligence Internship by Velocity Solutions.

---

## 📌 Overview

### Week 1:
Focused on building fundamental understanding of AI and implementing basic intelligent systems using Python. I practiced working with real datasets, rule-based systems, and classic search algorithms.

### Week 2:
Dived into Natural Language Processing (NLP) and text classification using machine learning. Built an end-to-end SMS Spam Detection system using two types of models.

### Week 3:
Explored image classification using traditional ML and CNN models on the MNIST handwritten digit dataset. Compared performance between classical algorithms (kNN, SVM) and deep learning.

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
- Applied rules to dataset rows using `apply()` and returned decisions like:
  - "Highly Recommended"
  - "Acceptable"
  - "Not Recommended"
- Built a rule-based agent to check if a Tic-Tac-Toe board is a winning state for 'X'.

#### 4. Search Algorithms (Maze Solvers)
- Implemented **Breadth-First Search (BFS)** and **Depth-First Search (DFS)** to solve mazes.
- BFS returns the shortest path.
- DFS explores all paths, visualizes dead ends.
- Visualized the maze grid using `"S"`, `"G"`, `"|"`, `"_"`, and `"#"`.

---

### Week 2

#### 1. Dataset Loading and Preprocessing
- Used the **SMS Spam Collection Dataset** from UCI ML Repository.
- Cleaned text by lowercasing, removing stopwords, punctuation, and applying stemming & lemmatization.
- Stored cleaned data in a `clean_text` column.

#### 2. Feature Extraction
- Transformed cleaned text into numerical features using:
  - **Bag-of-Words (CountVectorizer)**
  - **TF-IDF (TfidfVectorizer)**

#### 3. Model Training and Evaluation
- Trained two classifiers:
  - **Multinomial Naive Bayes** on BoW
  - **Logistic Regression** on TF-IDF
- Evaluated performance using accuracy, precision, recall, and F1-score.

| Model               | Vectorization | Accuracy | F1 (Spam) |
|---------------------|----------------|----------|-----------|
| Naive Bayes         | Bag-of-Words   | 96.95%   | 0.89      |
| Logistic Regression | TF-IDF         | 95.96%   | 0.83      |

#### 4. Real-Time Testing Interface
- Created an interactive input system where users can enter a message and receive predictions from both models.

#### 5. Model Comparison
- Compared recall, precision, and accuracy across both approaches.
- Observed how TF-IDF can help precision but may lower recall.

---

### Week 3

#### 1. Working with MNIST Dataset
- Loaded the MNIST handwritten digits dataset using `tensorflow.keras.datasets`.
- Visualized sample images using `matplotlib`.

#### 2. Data Preprocessing
- Normalized pixel values to range [0, 1].
- Flattened 28x28 images into vectors for use in traditional ML.

#### 3. ML Model Implementation
- Trained and evaluated:
  - **k-Nearest Neighbors (kNN)** → Accuracy: 94.63%
  - **Support Vector Machine (SVM)** → Accuracy: 88.87%
- Compared precision, recall, and F1-score across 10 digit classes.

#### 4. (Optional) CNN for Image Classification
- Built a **Convolutional Neural Network (CNN)** using TensorFlow/Keras:
  - Conv2D → MaxPooling → Flatten → Dense → Output
- Achieved ~98% accuracy on MNIST test set.
- Uploaded custom digit images and let the CNN predict the value.
- Displayed prediction **confidence scores** for each digit.

---

## 🧪 Technologies Used

- Python 3
- Google Colab
- Libraries: `pandas`, `nltk`, `sklearn`, `tensorflow`, `matplotlib`, `PIL`

---

## 📁 File Structure

- `Week1.ipynb` → Colab notebook for Week 1 tasks
- `Week2.ipynb` → Colab notebook for Week 2 tasks
- `Week3.ipynb` → Colab notebook for Week 3 tasks


