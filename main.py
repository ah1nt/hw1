import os
import pickle
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from search import grid_search
from train import train_model
from eval import evaluate_model, visualize_weights, error_analysis

def plot_curves(train_losses, val_losses, val_accuracies, save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_acc_curves.png'))
    plt.close()

def main():
    data_dir = 'EuroSAT_RGB'
    save_dir = 'output'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading Data...", flush=True)
    data = get_dataloaders(data_dir, batch_size=32)
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    classes = data['classes']
    
    input_dim = X_train.shape[1]
    num_classes = len(classes)
    
    print("Starting Grid Search...", flush=True)
    param_grid = {
        'lr': [0.05, 0.01],
        'hidden_dim': [64, 128],
        'weight_decay': [0.0, 0.001],
        'activation': ['relu', 'sigmoid'],
        'epochs': [5],
        'batch_size': [64]
    }
    
    best_params, _, results = grid_search(
        X_train, y_train, X_val, y_val, 
        input_dim=input_dim, num_classes=num_classes, param_grid=param_grid
    )
    
    # Save search results
    with open(os.path.join(save_dir, 'grid_search_results.txt'), 'w') as f:
        for res in results:
            f.write(f"Params: {res['params']}, Val Acc: {res['val_acc']:.4f}\n")
            
    print("Training Best Model...")
    epochs = 30
    best_model_res = train_model(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim, hidden_dim=best_params['hidden_dim'], num_classes=num_classes,
        activation=best_params['activation'], lr=best_params['lr'], weight_decay=best_params['weight_decay'],
        epochs=epochs, batch_size=best_params['batch_size'], verbose=True
    )
    
    best_weights = best_model_res['best_weights']
    model = best_model_res['model']
    
    # Save weights
    with open(os.path.join(save_dir, 'best_weights.pkl'), 'wb') as f:
        pickle.dump(best_weights, f)
        
    print("Plotting Curves...")
    plot_curves(best_model_res['train_losses'], best_model_res['val_losses'], best_model_res['val_accuracies'], save_dir)
    
    print("Evaluating Model...")
    preds, acc = evaluate_model(model, best_weights, X_test, y_test, classes, save_dir)
    
    print("Visualizing Weights...")
    visualize_weights(model, save_dir)
    
    print("Error Analysis...")
    error_analysis(X_test, y_test, preds, classes, save_dir, num_examples=5)

    print("Pipeline Complete! Results saved to 'output' directory.")

if __name__ == '__main__':
    main()
