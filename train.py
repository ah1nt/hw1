import numpy as np
from model import MLP, CrossEntropyLoss, SGDOptimizer
from dataset import batch_generator

def calculate_accuracy(model, X, y, batch_size=64):
    correct = 0
    total = 0
    for X_batch, y_batch in batch_generator(X, y, batch_size, shuffle=False):
        logits = model.forward(X_batch)
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    return correct / total

def train_model(X_train, y_train, X_val, y_val, 
                input_dim, hidden_dim, num_classes, 
                activation='relu', 
                lr=0.01, lr_decay=0.95, weight_decay=0.0001,
                epochs=50, batch_size=32, verbose=True):
    
    model = MLP(input_dim, hidden_dim, num_classes, activation)
    criterion = CrossEntropyLoss()
    optimizer = SGDOptimizer(model, lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        model_lr = optimizer.lr
        
        # Training
        epoch_train_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size, shuffle=True):
            # Forward
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            epoch_train_loss += loss
            
            # Backward
            optimizer.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            
            # Update
            optimizer.step()
            num_batches += 1
            
        epoch_train_loss /= num_batches
        train_losses.append(epoch_train_loss)
        
        # Validation
        val_loss = 0.0
        val_batches = 0
        for X_batch, y_batch in batch_generator(X_val, y_val, batch_size, shuffle=False):
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            val_loss += loss
            val_batches += 1
        
        val_loss /= val_batches
        val_losses.append(val_loss)
        
        val_acc = calculate_accuracy(model, X_val, y_val, batch_size)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | LR: {model_lr:.4f} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        # Learning Rate Decay
        optimizer.lr *= lr_decay
        
    return {
        'model': model,
        'best_weights': best_weights,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
