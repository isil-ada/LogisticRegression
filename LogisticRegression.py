import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, precision_score, recall_score, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

#Load dataset
df = pd.read_csv("Credit Card Defaulter Prediction.csv")

#Remove whitespace from column names
df.columns = df.columns.str.strip()

#Remove columns that are not useful for prediction 
df = df.drop(columns=["ID"] , axis=1)

#check for missing values
print(df.isnull().sum(),"\n")

print(df.info())

#identify categorical columns and convert them into numerical format using one-hot encoding
categorical_columns = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

print(df.info(),"\n")

#split features(x) and target-variable(y)
x = df.drop(columns=["default_Y"],axis=1)
y = df["default_Y"]

#split the dataset into training and test sets (%80 train - %20 test)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Standardize the feature values to improve model performance
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#training of the Logistic Regression model and training time
start_time_fit = time.perf_counter()
LR = LogisticRegression(max_iter=500)
model = LR.fit(x_train,y_train)
end_time_fit = time.perf_counter()
print(f"Training time of the model -> {end_time_fit-start_time_fit} second \n")

#prediction time
start_time_pred = time.perf_counter()
y_pred = model.predict(x_test)
end_time_pred = time.perf_counter()
print(f"Prediction time of the model -> {end_time_pred-start_time_pred} second \n")

#calculate model accuracy
accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
print(f"Accuracy of the model -> {accuracy} \n")

#calculate precision
precision = precision_score(y_true=y_test, y_pred=y_pred)
print(f"Precision of the model -> {precision} \n")

#calculate recall
recall = recall_score(y_true=y_test, y_pred=y_pred)
print(f"Recall of the model -> {recall} \n")

#calculate F1 score
f1 = f1_score(y_true=y_test, y_pred=y_pred)
print(f"F1 Score of the model -> {f1} \n")

#calculate log loss
y_prob = model.predict_proba(x_test)  #probabilities are required for log loss
loss = log_loss(y_true=y_test, y_pred=y_prob)
print(f"Log Loss of the model -> {loss} \n")

#calculate log loss values ​​for each sample
epsilon = 1e-15
y_prob = model.predict_proba(x_test)[:, 1]
y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
loss_values = -(y_test * np.log(y_prob) + (1 - y_test) * np.log(1 - y_prob))

#log loss graph
plt.figure(figsize=(8,5))
plt.plot(loss_values, marker='o', linestyle='', alpha=0.6)
plt.title("Sample-wise Log Loss Values", fontsize=15, fontweight="bold", fontfamily="Book Antiqua")
plt.xlabel("Sample Index", fontsize=13, fontweight="bold", fontfamily="Book Antiqua")
plt.ylabel("Log Loss", fontsize=13, fontweight="bold", fontfamily="Book Antiqua")
plt.grid(True)
plt.show()

#compute confussion matrix
matrix = confusion_matrix( y_pred=y_pred , y_true=y_test )
print(f"-Confusion Matrix- \n {matrix}")

#visualize the confussion matrix
plt.figure(figsize=(6.5,5))
sns.heatmap(matrix,fmt="d",cmap="Purples",annot=True,linewidths=2,linecolor="black")
plt.xlabel("Predicted Label",fontsize=17,fontweight="bold",fontfamily="Book Antiqua")
plt.ylabel("True Label",fontsize=17,fontweight="bold",fontfamily="Book Antiqua")
plt.title("CONFUSSION MATRIX",fontsize=20,fontweight="bold",fontfamily="Book Antiqua")
plt.show()


