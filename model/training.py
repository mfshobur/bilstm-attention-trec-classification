from collections import Counter
from typing import Union
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from alstm import ALSTMModel
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy as dc

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
            if early_stop:
                early_stop_counter = 0
            if log:
                print(f'Model saved with validation loss: {best_valid_loss:.4f}')
                print('early stopping counter restarted')
        else:
            # check for early stopping
            if early_stop:
                if epoch_valid_loss > best_valid_loss:
                    early_stop_counter += 1
                    if log:
                        print('early stopping counter:', early_stop_counter)
                    if early_stop_counter >= early_stop:
                        print('traning is stopped by early stopping on epoch:', epoch)
                        break
        
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

def log_class_distribution(labels, name):
    """Log the class distribution of the labels"""
    counter = Counter(labels)
    total = sum(counter.values())
    print(f"{name} class distribution (%):")
    for cls, count in sorted(counter.items()):
        pct = 100 * count / total
        print(f"  Class {cls}: {count} ({pct:.2f}%)")

def cross_validate(
        cfg,
        preprocessor,
        train_dataset,
        test_dataset,
        targets,
        criterion,
        learning_rate,
        weight_decay,
        device,
        collate_fn,
        label_names,
        best_model_path,
        fold=5,
        batch_size=32,
        num_epochs=5,
        debug=False
        ):
    
    history = {}
    best_valid_loss = float('inf')
    # best_valid_acc = 0
    best_model = None

    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    print(f'Total fold: {fold}')
    
    history['train'] = []

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_dataset, targets)):
        train_labels = [targets[i] for i in train_idx]
        val_labels = [targets[i] for i in val_idx]
        print(f"Fold: {fold_idx+1}:")
        log_class_distribution(train_labels, "Train")
        log_class_distribution(val_labels, "Validation")
        # print("Train class distribution:", Counter(train_labels))
        # print("Validation class distribution:", Counter(val_labels))
        # print(f"Train class distribution: {sum(targets[i] for i in train_idx)}")
        # print(f"Validation class distribution: {sum(targets[i] for i in val_idx)}")

        # create model
        model = create_model(cfg, preprocessor)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer_name = type(optimizer).__name__

        # create pytorch subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            collate_fn=collate_fn)
        
        new_fold = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=val_loader, 
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            best_model_path=best_model_path,
            log=debug,
            best_valid_loss=best_valid_loss
        )

        history['train'].append(new_fold)

        # update best valid loss
        fold_best_valid_loss = min(new_fold['valid_losses'])
        print(f'this fold best valid loss: {fold_best_valid_loss}\n', )
        if fold_best_valid_loss < best_valid_loss:
            best_valid_loss = fold_best_valid_loss
            best_model = dc(model)
            best_history = new_fold
            print(f'New model saved from fold {fold_idx+1}\n')

        # fold_best_valid_acc = max(new_fold['valid_accs'])
        # if fold_best_valid_acc < best_valid_acc:
        #     best_valid_acc = fold_best_valid_acc
        #     best_model = dc(model)
        #     print(f'New model saved from fold {fold+1}\n')
        
        del model
        del optimizer
            
    best_model_test_loss, best_model_test_acc, report, conf_mat = evaluate_model(
        model=best_model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        label_names=label_names,
        return_report=True
    )
    print(f"Final Test Accuracy (best model): {best_model_test_acc:.4f}")

    # save_to_excel(
    #     drop_rate=cfg['drop_rate'],
    #     test_acc=f'{best_model_test_acc:.4f}',
    #     test_loss=f'{best_model_test_loss:.4f}',
    #     vocab_size=preprocessor.vocab_size,
    #     n_heads=cfg['n_heads'],
    #     embedding_trainable=best_model.embedding.weight.requires_grad,
    #     # optimizer=type(optimizer).__name__,
    #     optimizer=optimizer_name,
    #     optimizer_lr=learning_rate,
    #     weight_decay=weight_decay,
    #     cv_fold=fold,
    #     epochs=num_epochs,
    #     use_stopwords=USE_STOPWORDS,
    #     notes='change feed forward linear into times 2'
    # )
    
    # return history, best_model
    return best_history, best_model_test_acc, best_model_test_loss, report, conf_mat, best_model

if __name__ == "__main__":
    pass