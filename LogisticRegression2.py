import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

class MyLogisticRegression :
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate  #alpha
        self.epochs = epochs  #num of iterations
        self.weights = None  
        self.bias = None  
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #train the Logistic Regression model using Gradient Descent
    def fit(self, x, y):
        num_samples, num_features = x.shape  #get data deimensions
        self.weights = np.zeros(num_features)  #initialize weights with zeros
        self.bias = 0  
        
        #update weights with Gradient Descent
        for _ in range(self.epochs):
            z = np.dot(x, self.weights) + self.bias  # z = wx + b
            predictions = self.sigmoid(z)  
            
            #calculate gradiens
            dw = (1 / num_samples) * np.dot(x.T, (predictions - y))  #Gradient for weights
            db = np.mean(predictions - y)  #Gradient for bias
            
            #update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db 

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        probability = self.sigmoid(z)
        return (probability >= 0.5).astype(int)  #convert probabilities to class labels (0 or 1)
    
    def accuracy(self,x,y) :
        predictions = self.predict(x)
        return np.mean(predictions == y)


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

#Standardize the feature values to improve model performance
scaler = StandardScaler()
x = scaler.fit_transform(x)

#split the dataset into training and test sets (%80 train - %20 test)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#training of the Logistic Regression model and training time
start_time_fit = time.perf_counter()
model = MyLogisticRegression(learning_rate=0.0099,epochs=450)
model.fit(x_train,y_train)
end_time_fit = time.perf_counter()
print(f"Training time of the model -> {end_time_fit-start_time_fit} second \n")

#prediction time
start_time_pred = time.perf_counter()
y_pred = model.predict(x_test)
end_time_pred = time.perf_counter()
print(f"Prediction time of the model -> {end_time_pred-start_time_pred} second \n")

#calculate model accuracy
print(f"Accuracy of the model -> {model.accuracy(x_test,y_test)} \n")

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



