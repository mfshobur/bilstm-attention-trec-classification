from typing import Union
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from alstm import ALSTMModel

def create_model(cfg, preprocessor, copy_embedding=True) -> ALSTMModel:
    """Create model on CPU. Make sure to move model to device"""
    model = ALSTMModel(cfg)
    if copy_embedding:
        if preprocessor.embedding_matrix is not None:
            print('copying embedding matrix')
            model.embedding.weight.data.copy_(torch.from_numpy(preprocessor.embedding_matrix))
            # make the embedding is not trainable
            model.embedding.weight.requires_grad = False
    return model

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[torch.device, str],
    num_epochs: int = 5,
    best_model_path='best_model.pt',
    best_valid_loss: float = float('inf'),
    log=True,
    early_stop=None,
    ):

    # Initialize best validation loss
    best_valid_loss = best_valid_loss
    
    # Training history
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    # early stopping
    if early_stop:
        early_stop_counter = 0
        print('early stopping implemented')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_samples = 0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            # print('token ids:', texts)
            predictions = model(texts)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Calculate accuracy
            predictions_class = torch.argmax(predictions, dim=1)
            correct = (predictions_class == labels).float().sum()
            
            # Update metrics
            epoch_train_loss += loss.item() * len(labels)
            epoch_train_acc += correct.item()
            train_samples += len(labels)
            
            if (batch_idx + 1) % 10 == 0 and log:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, ' 
                      f'Loss: {loss.item():.4f}, Acc: {correct.item()/len(labels):.4f}')
        
        # Calculate average loss and accuracy for the epoch
        epoch_train_loss /= train_samples
        epoch_train_acc /= train_samples
        
        # Validation
        model.eval()
        epoch_valid_loss = 0
        epoch_valid_acc = 0
        valid_samples = 0
        
        with torch.no_grad():
            for texts, labels in valid_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                # Forward pass
                predictions = model(texts)
                
                # Calculate loss
                loss = criterion(predictions, labels)
                
                # Calculate accuracy
                predictions_class = torch.argmax(predictions, dim=1)
                correct = (predictions_class == labels).float().sum()
                
                # Update metrics
                epoch_valid_loss += loss.item() * len(labels)
                epoch_valid_acc += correct.item()
                valid_samples += len(labels)
        
        # Calculate average validation loss and accuracy
        epoch_valid_loss /= valid_samples
        epoch_valid_acc /= valid_samples
        
        # Save the best model
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), best_model_path)
            if log:
                print(f'Model saved with validation loss: {best_valid_loss:.4f}')
        
        # check for early stopping
        if early_stop:
            if epoch_valid_loss > best_valid_loss:
                early_stop_counter += 1
                print('early stopping counter:', early_stop_counter)
                if early_stop_counter >= early_stop:
                    print('traning is stopped by early stopping')
                    break
            else:
                print('early stopping counter restarted')
                early_stop_counter = 0
        
        # Update history
        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)
        train_accs.append(epoch_train_acc)
        valid_accs.append(epoch_valid_acc)
        
        # Print epoch statistics
        if log:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
            print(f'Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_acc:.4f}')
            print('-' * 60)
    
    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs
    }

def evaluate_model(model, test_loader, criterion, device, label_names=None, return_report=False):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            predictions = model(texts)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Calculate accuracy
            predictions_class = torch.argmax(predictions, dim=1)
            correct = (predictions_class == labels).float().sum()
            
            # Collect predictions and labels for classification report
            all_predictions.extend(predictions_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update metrics
            test_loss += loss.item() * len(labels)
            test_acc += correct.item()
            test_samples += len(labels)
    
    # Calculate average test loss and accuracy
    test_loss /= test_samples
    test_acc /= test_samples
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # Print classification report
    if label_names is not None:
        print('\nClassification Report:')
        report = classification_report(all_labels, all_predictions, target_names=label_names, zero_division=0, output_dict=True)
        print(classification_report(all_labels, all_predictions, target_names=label_names, zero_division=0))
        print('\nConfusion Matrix:')
        confusion = confusion_matrix(all_labels, all_predictions)
        print(confusion)
    else:
        print('\nClassification Report:')
        print(classification_report(all_labels, all_predictions))
    
    if return_report:
        return test_loss, test_acc, report, confusion
    return test_loss, test_acc

if __name__ == "__main__":
    pass