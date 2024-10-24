from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
data = pd.read_csv('/content/Iris.csv')

X = data.drop('Species', axis=1)
y = data['Species']
# Split data into train and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
XtrainS = scaler.fit_transform(Xtrain)
XtestS = scaler.transform(Xtest)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
Xtrainp = pca.fit_transform(XtrainS)
Xtestp = pca.transform(XtestS)

# Train the Decision Tree classifier
DT = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_split=2)
DT.fit(Xtrainp, Ytrain)

# Predict and evaluate
y_pred = DT.predict(Xtestp)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(Ytest, y_pred))
