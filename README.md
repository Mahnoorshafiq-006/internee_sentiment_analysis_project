# internee_sentiment_analysis_project
# Sentiment Analysis of Internship Feedback

A machine learning project that classifies employee/intern feedback text as **Positive** or **Negative**, and identifies which words are most associated with negative feedback — helping highlight areas where satisfaction can be improved.

---

## 1. Project Objective

The goal of this project is to analyze written feedback about employees/interns and automatically determine whether the sentiment behind each review is **Positive** or **Negative**. Beyond just classification, the project also surfaces which specific words are most strongly linked to negative feedback, giving a starting point for identifying areas to improve.

**Task Requirements Addressed:**
- **Objective:** Analyze feedback to understand positive and negative sentiments
- **Model Type:** Sentiment Classification using Logistic Regression
- **Outcome:** Identify areas where satisfaction can be improved

---

## 2. Dataset

**Source:** Employee Review dataset (MTurk-generated, Kaggle)

**Original columns:**
| Column | Description |
|---|---|
| `id` | Unique record ID |
| `person_name` | Employee name |
| `nine_box_category` | One of 9 performance/potential categories (e.g., "Risk", "Star", "Core Player") |
| `feedback` | Free-text written review about the employee |
| `updated` | Whether the record was revised |
| `reviewed` | Whether the record has been reviewed |

**Size:** 225 feedback rows, covering 75 unique employees (each with multiple reviews).

### Why sentiment labels were derived, not provided
The original dataset does not contain a Positive/Negative label — it only contains 9 performance categories. Since no separately labeled sentiment dataset was available, a `sentiment` column was engineered from the existing data using this rule:

| Condition in `nine_box_category` | Assigned Sentiment |
|---|---|
| Contains "Low performance" | Negative |
| Contains "Moderate" or "High" performance | Positive |

This produced a working dataset of **150 Positive** and **75 Negative** examples.

---

## 3. Project Workflow (Step by Step)

1. **Load the dataset** — Read the CSV file into a table using `pandas`.
2. **Create sentiment labels** — Derive Positive/Negative from the performance level in `nine_box_category`.
3. **Clean the feedback text** — Lowercase the text and strip punctuation so the model treats words consistently (e.g., "Good." and "good" are treated the same).
4. **Convert text to numbers (TF-IDF)** — Machine learning models can't read words directly, so `TfidfVectorizer` converts each review into a vector of word-importance scores.
5. **Split into train/test sets** — 80% of the data is used to train the model, 20% is held out to test it, keeping the same Positive/Negative ratio in both (`stratify=y`).
6. **Train the model** — A `LogisticRegression` classifier (with `class_weight="balanced"` to handle the 2:1 class imbalance) learns the relationship between word patterns and sentiment.
7. **Evaluate performance** — Accuracy, precision, recall, and F1-score are calculated on the unseen test set.
8. **Identify improvement areas** — The model's learned word weights are ranked to show the top words most associated with Negative feedback (and, separately, Positive feedback).
9. **Test on a new review** — A sample sentence is run through the full pipeline to demonstrate the model working on unseen text.
10. **Save predictions to CSV** — The trained model predicts sentiment for every row in the dataset, and the results are saved to a new CSV file alongside the original data, for review and reporting.

---

## 4. Tools & Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Loading and manipulating the dataset |
| `scikit-learn` (`sklearn`) | TF-IDF vectorization, train/test split, Logistic Regression model, evaluation metrics |

---

## 5. Results

- **Model:** Logistic Regression with TF-IDF features
- **Accuracy:** ~80% on held-out test data
- **Negative class performance:** ~71% precision, ~67% recall
- **Positive class performance:** ~84% precision, ~87% recall

### Words most linked to Negative feedback (areas to improve):
Words such as *"not"*, *"lot"*, *"yet"* — often appearing in phrases describing incomplete work, inconsistency, or unmet expectations.

### Words most linked to Positive feedback (strengths):
Words such as *"always"*, *"great"*, *"high"* — typically appearing in descriptions of reliability and strong output.

---

## 6. Known Limitations

- **Bag-of-words limitation:** The model (TF-IDF + Logistic Regression) evaluates individual word frequency, not sentence structure or word order. As a result, it can misclassify sentences where a strong positive word appears in a negative context (e.g., "He is **always** late" may be misread as Positive because of the word "always" alone, despite the overall sentence being negative).
- **Small dataset:** With only 75 unique employees, the model has limited examples to learn nuanced language patterns from.
- **Labels are derived, not original:** Since the dataset didn't include true sentiment labels, the Positive/Negative split is a reasonable approximation based on performance level, not manually verified ground truth.
- **Future improvement:** A context-aware model such as a Transformer (e.g., BERT) would better handle negation and word order, and is the natural next step beyond this baseline.

---

## 7. How to Run This Project

1. Make sure the dataset file `employee_review_dataset.csv` is in the same folder as the script.
2. Install the required libraries:
   ```
   pip install pandas scikit-learn
   ```
3. Run the script:
   ```
   python sentiment_classifier.py
   ```
4. Review the printed output (accuracy, classification report, top words) and check the generated file `employee_review_with_predictions.csv` for row-by-row predictions.

---

## 8. Output Files

| File | Description |
|---|---|
| `sentiment_classifier.py` | Main script — trains and evaluates the sentiment classification model |
| `employee_review_with_predictions.csv` | Original dataset with two added columns: `sentiment` (derived label) and `predicted_sentiment` (model's prediction) |

---

## 9. Conclusion

This project demonstrates a complete, working sentiment classification pipeline: from raw text feedback to a trained model capable of identifying sentiment and surfacing specific language patterns tied to negative feedback. While built on a simple, interpretable model suited for a small dataset, the project also identifies clear next steps (contextual models like Transformers) for improving accuracy on nuanced language.
