from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv('/content/Social_Network_Ads.csv')
data.head()
data.info()
print(data.isnull().sum())
data.fillna(0,inplace=True)
data['Gender']=data['Gender'].apply(lambda x:1 if x == 'Female' else 0)
x = data[['Gender','Age','EstimatedSalary']].values
y= data['Purchased'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import numpy as np

linear_svm=SVC(kernel='linear')
linear_svm.fit(x_train,y_train)
y_pred = linear_svm.predict(x_test)
print(y_pred)
def predctionop(svm,gender,age,salary):
  input_data=np.array([[gender,age,salary]])
  input_data_scaled=scaler.transform(input_data)
  pred=svm.predict(input_data_scaled)
  return 'Yes' if pred == 1 else 'No'
# Confusion Matrix for Linear Kernel SVM
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
print("Confusion Matrix (Linear Kernel):\n", conf_matrix_linear)

# Evaluate the model
accuracy_linear = accuracy_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

print(f"Accuracy (Linear Kernel): {accuracy_linear}")
print(f"Recall (Linear Kernel): {recall_linear}")
print(f"Precision (Linear Kernel): {precision_linear}")
print(f"F1-Measure (Linear Kernel): {f1_linear}")
