import itertools
from train import train_model

def grid_search(X_train, y_train, X_val, y_val, input_dim, num_classes, param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    best_acc = 0.0
    best_params = None
    best_model_state = None
    
    results = []
    
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        print(f"--- Running Config {i+1}/{len(combinations)}: {params} ---")
        
        # Unpack parameters
        hidden_dim = params.get('hidden_dim', 128)
        lr = params.get('lr', 0.01)
        weight_decay = params.get('weight_decay', 0.0001)
        activation = params.get('activation', 'relu')
        
        # Keep epochs and batch_size small for faster search
        epochs = params.get('epochs', 10)
        batch_size = params.get('batch_size', 64)
        
        result = train_model(
            X_train, y_train, X_val, y_val,
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            activation=activation, lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size, verbose=False
        )
        
        val_acc = result['best_val_acc']
        print(f"Validation Accuracy for config {i+1}: {val_acc:.4f}")
        
        results.append({
            'params': params,
            'val_acc': val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_model_state = result
            
    print("=== Grid Search Completed ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best Parameters: {best_params}")
    
    return best_params, best_model_state, results
