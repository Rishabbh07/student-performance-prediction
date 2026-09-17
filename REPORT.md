# Student Performance Prediction Using Machine Learning

**Course:** Fundamentals of Artificial Intelligence and Machine Learning  
**Student:** Rishabh Patel  
**Registration No.:** 25MIM10131  
**Branch:** CSE  
**College:** VIT Bhopal University

## Abstract

This project predicts a student's possible Pass/Fail result using study hours and attendance. A Decision Tree Classifier is trained on a small synthetic dataset and executed through the command line.

## Problem Statement

Create a supervised machine learning system that classifies a student as Pass or Fail from study hours and attendance percentage.

## Objectives

- Prepare a tabular dataset.
- Train a Decision Tree Classifier.
- Evaluate predictions using accuracy and a classification report.
- Build a terminal-executable project.

## Methodology

1. Create the dataset.
2. Separate features and target.
3. Split data into training and testing sets.
4. Train the model.
5. Evaluate it.
6. Predict a new student's result.

## Algorithm

A Decision Tree Classifier divides data using feature-based conditions and assigns a class at the final leaf. The depth is limited to reduce complexity.

## Execution

```bash
python student_performance.py --study-hours 6 --attendance 80
```

## Result and Limitations

The application prints evaluation metrics and a prediction. The dataset is synthetic and small; therefore, the results are only for learning and are not suitable for real academic decisions.

## Future Scope

Use a larger ethical dataset, add more features, compare algorithms, apply cross-validation, and develop a validated interface.

## References

- https://docs.python.org/3/
- https://pandas.pydata.org/docs/
- https://scikit-learn.org/stable/
