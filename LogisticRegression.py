import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
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


