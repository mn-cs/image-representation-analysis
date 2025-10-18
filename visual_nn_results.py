import matplotlib.pyplot as plt

def visualize_nn_results(classifier, x_train, y_train, x_test, y_test, feature_name=""):
    """
    Visualize correctly and incorrectly classified images with their nearest neighbors
    
    Args:
        classifier: trained KNN classifier
        x_train: training images (N, C, H, W)
        y_train: training labels
        x_test: test images (N, C, H, W) 
        y_test: test labels
        feature_name: name of the feature representation for display
    """
    # Get predictions
    predictions = classifier.predict(classifier._fit_X)  # Use internal training features
    
    # Find correctly and incorrectly classified test samples
    correct_indices = []
    incorrect_indices = []
    
    for i in range(len(y_test)):
        if len(correct_indices) < 5 and predictions[i] == y_test[i]:
            correct_indices.append(i)
        elif len(incorrect_indices) < 5 and predictions[i] != y_test[i]:
            incorrect_indices.append(i)
        
        if len(correct_indices) == 5 and len(incorrect_indices) == 5:
            break
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Plot correctly classified images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Correctly Classified Images ({feature_name})', fontsize=16)
    
    for i, idx in enumerate(correct_indices):
        # Find nearest neighbor
        distances, nn_indices = classifier.kneighbors([classifier._fit_X[idx]], n_neighbors=1)
        nn_idx = nn_indices[0][0]
        
        # Plot test image
        test_img = x_test[idx].transpose(1, 2, 0)
        axes[0, i].imshow(test_img)
        axes[0, i].set_title(f'Test: {class_names[y_test[idx]]}')
        axes[0, i].axis('off')
        
        # Plot nearest neighbor
        nn_img = x_train[nn_idx].transpose(1, 2, 0)
        axes[1, i].imshow(nn_img)
        axes[1, i].set_title(f'NN: {class_names[y_train[nn_idx]]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot incorrectly classified images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Incorrectly Classified Images ({feature_name})', fontsize=16)
    
    for i, idx in enumerate(incorrect_indices):
        # Find nearest neighbor
        distances, nn_indices = classifier.kneighbors([classifier._fit_X[idx]], n_neighbors=1)
        nn_idx = nn_indices[0][0]
        
        # Plot test image
        test_img = x_test[idx].transpose(1, 2, 0)
        axes[0, i].imshow(test_img)
        axes[0, i].set_title(f'Test: {class_names[y_test[idx]]}')
        axes[0, i].axis('off')
        
        # Plot nearest neighbor
        nn_img = x_train[nn_idx].transpose(1, 2, 0)
        axes[1, i].imshow(nn_img)
        axes[1, i].set_title(f'NN: {class_names[y_train[nn_idx]]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()