import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# load datasets
iris= load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['species']= iris.target

# feature and target
X= df[iris.feature_names]
y= df['species']

# train/test split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

# train MLP model
model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42)
model.fit(X_train,y_train)

# prediction
y_pred= model.predict(X_test)

# evaluation
print("accuracy ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot loss curve
plt.plot(model.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()