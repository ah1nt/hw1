import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

def evaluate_model(model, weights, X_test, y_test, classes, save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Load best weights
    model.set_weights(weights)
    
    # Predict
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)
    
    # Accuracy
    acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Classification Report (Precision, Recall, F1)
    report = classification_report(y_test, preds, target_names=classes)
    print("Classification Report:")
    print(report)
    
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return preds, acc

def visualize_weights(model, save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)
    weights = model.layers[0].W  # Shape: (input_dim, hidden_dim)
    # Assume input_dim is 64*64*3 = 12288
    # We want to visualize some hidden units
    hidden_dim = weights.shape[1]
    
    num_to_plot = min(16, hidden_dim)  # Plot up to 16 hidden units
    grid_size = int(np.ceil(np.sqrt(num_to_plot)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(num_to_plot):
        w = weights[:, i]
        # Normalize to [0, 1] for visualization
        w_min, w_max = w.min(), w.max()
        if w_max - w_min > 0:
            w_img = (w - w_min) / (w_max - w_min)
        else:
            w_img = w
        
        w_img = w_img.reshape(64, 64, 3)
        
        axes[i].imshow(w_img)
        axes[i].axis('off')
        axes[i].set_title(f'Unit {i+1}')
        
    for j in range(num_to_plot, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weights_visualization.png'))
    plt.close()

def error_analysis(X_test, y_test, preds, classes, save_dir='output', num_examples=5):
    os.makedirs(save_dir, exist_ok=True)
    
    errors = np.where(preds != y_test)[0]
    if len(errors) == 0:
        print("No errors found!")
        return
        
    np.random.shuffle(errors)
    errors = errors[:num_examples]
    
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    if num_examples == 1:
        axes = [axes]
        
    for idx, err_idx in enumerate(errors):
        img_flat = X_test[err_idx]
        true_label = classes[y_test[err_idx]]
        pred_label = classes[preds[err_idx]]
        
        img = img_flat.reshape(64, 64, 3)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'))
    plt.close()

    plt.close()
