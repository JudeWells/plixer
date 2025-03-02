import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
import wandb

if __name__ == "__main__":
    paths = {
        "train_x": "outputs/aa_classifier_data_5k/train_embeddings.pt",
        "train_y": "outputs/aa_classifier_data_5k/train_labels.pt",
        "val_x": "outputs/aa_classifier_data_5k/val_embeddings.pt",
        "val_y": "outputs/aa_classifier_data_5k/val_labels.pt",
        "test_x": "outputs/aa_classifier_data_5k/test_embeddings.pt",
        "test_y": "outputs/aa_classifier_data_5k/test_labels.pt",
    }

    # Load all datasets
    train_x = torch.load(paths["train_x"])
    train_y = torch.load(paths["train_y"])
    val_x = torch.load(paths["val_x"])
    val_y = torch.load(paths["val_y"])
    test_x = torch.load(paths["test_x"])
    test_y = torch.load(paths["test_y"])

    # Convert to numpy arrays for sklearn
    train_x_np = train_x.numpy()
    train_y_np = train_y.numpy()
    val_x_np = val_x.numpy()
    val_y_np = val_y.numpy()
    test_x_np = test_x.numpy()
    test_y_np = test_y.numpy()

    # Initialize Gaussian Process Classifier
    kernel = 1.0 * RBF(length_scale=1.0)  # Basic RBF kernel
    gp_classifier = GaussianProcessClassifier(
        kernel=kernel,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Initialize WandB logging
    wandb.init(project="amino_acid_classification_demo", name="gp_classifier")
    
    print("Training Gaussian Process classifier...")
    gp_classifier.fit(train_x_np, train_y_np)
    print("Training complete")
    # Evaluate on all datasets
    for name, X, y in [("train", train_x_np, train_y_np),
                       ("val", val_x_np, val_y_np),
                       ("test", test_x_np, test_y_np)]:
        y_pred = gp_classifier.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"{name.capitalize()} Accuracy: {acc:.4f}")
        wandb.log({f"{name}_accuracy": acc})