import numpy as np
import matplotlib.pyplot as plt


def visualize_cifar_data(images, labels, samples_per_class=6):
    """Display sample CIFAR-10 images per class."""

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(cifar_classes)

    plt.rcParams['figure.figsize'] = (20.0, 16.0)

    for cls_index, cls_name in enumerate(cifar_classes):
        idxs = np.flatnonzero(labels == cls_index)
        selected_idxs = np.random.choice(idxs, samples_per_class, replace=False)

        for i, idx in enumerate(selected_idxs):
            plt_idx = i * num_classes + cls_index + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(images[idx] / 255.0)
            plt.axis('off')
            if i == 0:
                plt.title(cls_name)

    plt.show()


def visualize_nearest_neighbors(x_test, y_test, x_train, y_train, knn_classifier, test_features, feature_name=""):
    """Visualize correctly and incorrectly classified test images along with their nearest neighbors from the training set."""

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']

    predicted_labels = knn_classifier.predict(test_features)
    neighbor_indices = knn_classifier.kneighbors(test_features, n_neighbors=1, return_distance=False).flatten()

    correct_idxs   = np.where(predicted_labels == y_test)[0][:5]
    incorrect_idxs = np.where(predicted_labels != y_test)[0][:5]

    for idxs, title in [
        (correct_idxs, f"Correctly Classified Images ({feature_name})"),
        (incorrect_idxs, f"Incorrectly Classified Images ({feature_name})")
    ]:
        plt.figure(figsize=(15, 6))
        plt.suptitle(title, fontsize=14)
        for col, idx in enumerate(idxs):
            plt.subplot(2, 5, col + 1)
            plt.imshow(x_test[idx].transpose(1, 2, 0) / 255.0)
            plt.title(f"Test: {cifar_classes[y_test[idx]]}", fontsize=8)
            plt.axis('off')

            nn_idx = neighbor_indices[idx]
            plt.subplot(2, 5, 5 + col + 1)
            plt.imshow(x_train[nn_idx].transpose(1, 2, 0) / 255.0)
            plt.title(f"NN: {cifar_classes[y_train[nn_idx]]}", fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.show()