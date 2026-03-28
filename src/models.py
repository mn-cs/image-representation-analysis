from sklearn.neighbors import KNeighborsClassifier

def run_nearest_neighbor(x_train, y_train, x_test, y_test):
    """Train and evaluate a 1-NN classifier."""
    nn_classifier = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
    
    nn_classifier.fit(x_train, y_train)

    test_acc = nn_classifier.score(x_test, y_test)
    print(f"Nearest neighbor accuracy on the test set: {test_acc:.6f}")
    
    return nn_classifier