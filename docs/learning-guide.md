# Learning guide

## 1. What is supervised learning?

Each training example consists of an input `x` and a correct answer `y`:

```text
x = 64 grayscale pixel values from one 8x8 image
y = the digit written in that image, such as 7
```

The model learns parameters that make the correct label more likely. During prediction, it receives only the image. The correct label is used later to evaluate its answer.

## 2. Why flatten an image?

This baseline treats each pixel position as a separate numeric feature:

```text
(8, 8) image -> (64,) feature vector
N images    -> (N, 64) matrix
N labels    -> (N,) vector
```

Flattening discards the explicit two-dimensional structure. A convolutional neural network can use spatial structure more directly, but a linear classifier is a smaller starting point for understanding the workflow.

## 3. What does the model calculate?

After standardization, each possible class `k` receives a score:

```text
score[k] = dot(weights[k], x) + bias[k]
```

For this ten-class problem, multinomial logistic regression turns the scores into probabilities with softmax:

```text
probability[k] = exp(score[k]) / sum(exp(score[j]) for j in classes)
```

Training minimizes a regularized classification loss. The contribution of a single labeled example to the unregularized loss is:

```text
loss = -log(probability[correct_label])
```

If the model gives the correct label a low probability, the loss is larger. scikit-learn's optimizer updates weights and biases to reduce the objective. This repository uses its implementation rather than writing an optimizer from scratch.

## 4. Why fit preprocessing only on training data?

`StandardScaler` estimates each pixel feature's mean and standard deviation from the training images. The fitted transformation is then applied to validation and test images.

If you fit the scaler before splitting the dataset, information from held-out images influences preprocessing. The `Pipeline` in `model.py` keeps preprocessing and the classifier together, and `fit` receives only the training matrix.

## 5. Why have three splits?

| Split | Purpose | Used to fit weights? |
| --- | --- | --- |
| Training, about 60% | Learn preprocessing and classifier parameters | Yes |
| Validation, about 20% | Compare settings such as regularization | No |
| Test, about 20% | Estimate performance after choosing settings | No |

Stratification approximately preserves the digit proportions in each split. The seed makes the split repeatable within the same software environment. This is a sample-level split, not a writer-held-out evaluation, so the results should not be interpreted as a guarantee for handwriting from new people or devices.

## 6. How should you read the results?

**Accuracy** is the number of correct predictions divided by the number of images. Read training and validation accuracy together. Very high training accuracy with lower validation accuracy can suggest overfitting.

The **confusion matrix** has true labels on the rows and predicted labels on the columns. Diagonal entries are correct predictions. An entry in row 3, column 8 counts images labeled 3 that the model predicted as 8.

`mistakes.png` helps connect numeric errors to actual images. Some low-resolution digits are ambiguous. Inspecting these examples can teach more than focusing on one accuracy number.

`classification_report` includes per-class precision, recall, F1 score, and support. A strong overall accuracy can still hide one digit that is harder to recognize.

## 7. Suggested exercises

1. Inspect sample images and explain what one row of `x_train` contains.
2. Find the exact line where the model receives the labels during fitting.
3. Compare `C=0.1`, `C=1`, and `C=10` using the same seed and validation split.
4. Remove the scaler temporarily and compare validation results.
5. Read ten mistakes and identify recurring confusions.
6. Replace logistic regression with a small `MLPClassifier` while preserving the split and evaluation.
7. Try a hand-drawn digit and inspect how resizing to 8x8 changes it.

Record the seed, model settings, software versions, and validation result for each experiment. Do not present the best test score from repeated trials as an unbiased final result.

## References

- [scikit-learn: load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [scikit-learn: recognizing handwritten digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)

No example accuracy in this guide is a measured result from this repository; execution was intentionally skipped when it was created.
