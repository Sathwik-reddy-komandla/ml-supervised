from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, random_state=42)
print(X)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train a Decision Tree model using Scikit-learn
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)

# Evaluate the model
test_acc = tree_classifier.score(X_test, y_test)
print(f'Test accuracy: {test_acc}')
