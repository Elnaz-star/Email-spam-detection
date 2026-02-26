# Email-spam-detection
# Spam Email Detection using Weighted KNN from Scratch

This project implements a **Spam Email Detection system** using a **custom Weighted K-Nearest Neighbors (KNN) algorithm** built entirely from scratch in Python. The goal is to classify emails as either *spam* or *ham (non-spam)* based on their content.

---

## Features

- **Custom KNN Implementation**:  
  The core algorithm is implemented manually without using pre-built KNN functions from libraries.  
  It uses **weighted voting**, where closer neighbors have a higher influence on the predicted class.

- **Text Preprocessing**:  
  Emails are cleaned and normalized with the following steps:  
  - Lowercasing all text  
  - Replacing email addresses with `emailaddress`  
  - Replacing URLs with `url`  
  - Replacing numbers with `number`  
  - Removing special characters and extra spaces  

- **TF-IDF Vectorization**:  
  Text data is converted into numerical vectors using **TF-IDF**, capturing the importance of each word in the corpus.  
  This allows the KNN algorithm to work with the email content effectively.

- **Normalization**:  
  All feature vectors are normalized to ensure fair distance calculations in high-dimensional space.

- **Distance Metric**:  
  Uses **cosine distance** to measure similarity between emails.

- **Weighted Voting**:  
  Instead of simple majority voting, the prediction considers the **inverse of the distance** of each neighbor, giving more weight to closer neighbors.

- **Model Evaluation**:  
  Accuracy is calculated, and a **Confusion Matrix** is generated to visualize the performance and error distribution of the model.

- **Predicting New Emails**:  
  The system can classify new, unseen emails after preprocessing and vectorization.

---

## Example Usage

```python
new_email = "hello elnaz, it is sara, call me please"
new_email_vec = vectorize.transform([new_email])
new_email_vec = normalize(new_email_vec).toarray()

prediction = knn_predict_weighted(X_train_vec.toarray(), y_train.values, new_email_vec, k=5)
print("Predicted class:", prediction)
