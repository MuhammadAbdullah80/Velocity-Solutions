# Velocity-Solutions

## AI Internship – Week 1 to Week 4 Tasks

This repository contains all my work from **Week 1**, **Week 2**, **Week 3**, and **Week 4** of the Artificial Intelligence Internship by Velocity Solutions.

---

## Overview

### Week 1:
Built foundational understanding of AI concepts and implemented basic intelligent agents using Python. Practiced working with real datasets, rule-based decision systems, and search algorithms.

### Week 2:
Focused on Natural Language Processing (NLP) and text classification. Created an SMS Spam Detection pipeline using two different ML models and evaluated their performance.

### Week 3:
Worked on image classification using both traditional ML and deep learning. Trained models on the MNIST handwritten digits dataset and compared kNN, SVM, and CNN performances.

### Week 4:
Developed a complete deep learning pipeline for detecting whether a person is wearing glasses using facial images. Trained a custom CNN model and integrated it with OpenCV for testing and real-time detection.

---

## Tasks Completed

### Week 1

#### 1. Understanding AI, ML, and DL
- Compared Artificial Intelligence, Machine Learning, and Deep Learning with examples and differences.
- Summarized key concepts and applications in a markdown cell.

#### 2. Data Exploration and Pandas Practice
- Loaded and explored two datasets: Car Evaluation and Tic-Tac-Toe Endgame.
- Performed `.head()`, `.describe()`, `.value_counts()`, and applied conditional filtering.
- Cleaned and understood the data using Pandas operations.

#### 3. Rule-Based AI Agents
- Built a rule-based recommendation system to classify car evaluation as "Highly Recommended", "Acceptable", or "Not Recommended".
- Created a Tic-Tac-Toe game state checker to detect whether 'X' has won using logic-based rules.

#### 4. Search Algorithms (Maze Solver)
- Implemented Breadth-First Search (BFS) and Depth-First Search (DFS) algorithms to solve maze problems.
- Visualized the search process and explained how each algorithm works.

---

### Week 2

#### 1. Text Preprocessing and Cleaning
- Used the SMS Spam Collection Dataset.
- Cleaned and tokenized text data using NLTK.
- Applied stopword removal, punctuation filtering, stemming, and lemmatization.

#### 2. Feature Extraction
- Transformed text into numerical features using:
  - CountVectorizer (Bag-of-Words)
  - TfidfVectorizer (TF-IDF)

#### 3. Model Training and Evaluation
- Trained two classifiers:
  - Multinomial Naive Bayes on Bag-of-Words
  - Logistic Regression on TF-IDF
- Evaluated with accuracy, precision, recall, and F1-score.

| Model               | Vectorization | Accuracy | F1 (Spam) |
|---------------------|----------------|----------|-----------|
| Naive Bayes         | Bag-of-Words   | 96.95%   | 0.89      |
| Logistic Regression | TF-IDF         | 95.96%   | 0.83      |

#### 4. Real-Time User Testing
- Built a system to take user input and classify it as "Spam" or "Ham" using both models.

---

### Week 3

#### 1. MNIST Dataset and Preprocessing
- Loaded the MNIST handwritten digits dataset.
- Normalized pixel values and visualized sample digits using Matplotlib.

#### 2. Traditional Machine Learning
- Flattened images into vectors and trained:
  - k-Nearest Neighbors (kNN) with ~94.6% accuracy.
  - Support Vector Machine (SVM) with ~88.9% accuracy.

#### 3. Deep Learning with CNN
- Built a CNN using TensorFlow:
  - Conv2D → MaxPooling → Flatten → Dense
- Trained model achieved ~98% test accuracy.
- Tested model on uploaded handwritten digits and visualized confidence scores.

---

### Week 4

#### 1. Problem Definition
Proposed and solved a real-world image classification problem:
- Objective: Detect whether a person is wearing glasses or not from an image.
- Use Cases:
  - Single image prediction
  - Group photo glasses count
  - Real-time webcam-based detection

#### 2. Dataset Description
- Used the "Glasses or No Glasses" dataset by Jeff Heaton.
- Included 5,000 facial images (`face-1.jpg` to `face-5000.jpg`).
- Provided `train.csv` (1–4500) and `test.csv` (4501–5000) for training and testing respectively.
- Each image labeled as either 1 (glasses) or 0 (no glasses).

#### 3. Initial Approach with Haarcascade
- Used OpenCV Haarcascade classifiers for face and eye/glasses detection.
- Added padding around eye regions to mimic glasses bounding boxes.
- Challenges:
  - Haarcascade detected eyes, not glasses.
  - False positives in cases without glasses.
- Result: Quick baseline solution, but limited real-world accuracy.

#### 4. Real-Time Detection Using Webcam (Local Only)
- Implemented face and glasses detection using webcam feed.
- Displayed results with red rectangles and label texts.
- Limitation: Google Colab cannot access webcams, so testing was done locally.

#### 5. Custom CNN Model for Glasses Detection
- Built a CNN using TensorFlow/Keras with:
  - 3 convolutional blocks (Conv2D → BatchNorm → MaxPooling)
  - Dense and Dropout layers
  - Binary output with sigmoid activation
- Trained the model on resized 100x100 images from `train.csv`.
- Model saved as `glasses_cnn_model.h5`.

#### 6. Obstacles and Fixes
- Windows path errors (`\V` interpreted as escape sequence):
  - Fixed using raw string literals (`r"path"`).
- Missing column names (`file` expected instead of `id`):
  - Fixed by generating filenames from the `id` column.
- Shape mismatch during inference in Colab:
  - Training used (100,100) images, but testing used different size.
  - Re-trained model with standardized input shape (64x64).
- Haarcascade glasses box not aligning:
  - Used bounding box merging + dynamic padding for cleaner results.
  - Label text sometimes duplicated or missing—solved with conditional placement logic.

#### 7. Integration in Google Colab
- Uploaded image for testing.
- Preprocessed image to match model input size.
- Predicted label using trained CNN.
- Used Haarcascade only to highlight glasses region if model prediction was positive.
- Displayed final image with red box and label.

#### 8. Final Webcam App with CNN (Local Only)
- Used webcam frames in real-time.
- Detected face region using Haarcascade.
- Cropped and resized face to feed into the CNN.
- If CNN predicted “Wearing Glasses”, added a red rectangle and label.
- Exit via pressing 'q'.

---

## Technologies Used

- Python 3
- Google Colab and PyCharm (local)
- OpenCV
- TensorFlow and Keras
- Scikit-learn
- NLTK
- Pandas and Matplotlib
- Haarcascade classifiers

---

## File Structure

- `Week1.ipynb` → All tasks from Week 1  
- `Week2.ipynb` → NLP and text classification  
- `Week3.ipynb` → MNIST + CNN vs ML  
- `Week4.ipynb` → Glasses Detection System using CNN and OpenCV  
- `glasses_cnn_model.h5` → Trained CNN model file for glasses classification  
- `haarcascade_eye_tree_eyeglasses.xml` → Haarcascade used for eye/glasses detection
- `Glasses.py` → Python script for real-time glasses detection via webcam (runs locally)
- `Project Documentation.pdf` → Complete documentation of the Week 4 project, including problem statement, dataset, model pipeline, challenges, and results

---

## Drive Links to Notebooks

- [Week 1](https://colab.research.google.com/drive/12ucA5eNC0mR4TXsNzmI6VAGtrhJ3Bfzx?usp=drive_link)
- [Week 2](https://colab.research.google.com/drive/1abW4Dt4EBVtwXn0yVCWZqb8gKrSscrRD?usp=drive_link)
- [Week 3](https://colab.research.google.com/drive/1Pg53DqeishP7PKC12YCTmuHPppqG3xsb?usp=drive_link)
- [Week 4](https://colab.research.google.com/drive/1wruhK3Qe0UjkzySbEO681u_p5ZGmvt81?usp=drive_link)

---


## Summary

Throughout the 4 weeks, I worked on a wide range of AI applications including rule-based systems, search algorithms, NLP pipelines, machine learning models, and deep learning-based image classification. In the final project, I successfully built and deployed a glasses detection system, tackled multiple real-world issues during implementation, and gained hands-on experience in model training, testing, and computer vision.

