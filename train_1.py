from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target
# Load and prepare data
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def label_to_class(label):
    if label == 0:
        return "benign"
    else:
        return "malignant"
clf = RandomForest(n_trees=300)
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)
print(predictions)

# Load and make predictions on new data

dtypes = {
    'id': str,
    'diagnosis': str,
    'radius_mean': float,
    'texture_mean': float,
    'perimeter_mean': float,
    'area_mean': float,
    'smoothness_mean': float,
    'compactness_mean': float,
    'concavity_mean': float,
    'concave points_mean': float,
    'symmetry_mean': float,
    'fractal_dimension_mean': float,
    'radius_se': float,
    'texture_se': float,
    'perimeter_se': float,
    'area_se': float,
    'smoothness_se': float,
    'compactness_se': float,
    'concavity_se': float,
    'concave points_se': float,
    'symmetry_se': float,
    'fractal_dimension_se': float,
    'radius_worst': float,
    'texture_worst': float,
    'perimeter_worst': float,
    'area_worst': float,
    'smoothness_worst': float,
    'compactness_worst': float,
    'concavity_worst': float,
    'concave points_worst': float,
    'symmetry_worst': float,
    'fractal_dimension_worst': float
}
new_data = pd.read_csv('test_predict_dataset.csv', dtype=dtypes)

X_new = new_data.drop('diagnosis', axis=1).values.astype(float)


new_predictions = clf.predict(X_new)

print("Predictions for new data:")
for pred in new_predictions:
    class_label = label_to_class(pred)
    print(f"Tumor is {class_label}")