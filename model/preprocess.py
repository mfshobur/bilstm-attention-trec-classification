import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

def collate_fn(batch):
    """Collate function to pad sequences in a batch"""
    texts, labels = zip(*batch)
    
    # Pad sequences to the length of the longest sequence in the batch
    padded_texts = pad_sequence([text for text in texts], batch_first=True, padding_value=0)
    
    return padded_texts.to(torch.int64), torch.tensor(labels)

class TextClassificationDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])
    
def read_dataset(path: str, google_drive=False) -> pd.DataFrame:
  if google_drive:
    from google.colab import drive
    drive.mount('/content/drive')
    df = pd.read_csv(f'/content/drive/{path}')
    return df
  else:
    df = pd.read_csv(path)
    return df
  
def log_model_train(
        n_heads,
        drop_rate,
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        best_model_test_acc,
        best_model_test_loss,
        time,
        file='alstm_model.csv',
    ):
    columns = {
        'no': 0,
        'n_heads': str(n_heads),
        'drop_rate': str(drop_rate),
        'train_acc': str(train_acc),
        'test_acc': str(test_acc),
        'train_loss': str(train_loss),
        'test_loss': str(test_loss),
        'best_model_test_acc': str(best_model_test_acc),
        'best_model_test_loss': str(best_model_test_loss),
        'time': str(time)
    }

    try:
        df = pd.read_csv(file)
        last_index = df.shape[0]+1
        columns['no'] = str(last_index)
        
        text = ''
        text += f'\n{",".join(columns.values())}'

        with open(file, 'a') as f:
            f.write(text)
            f.close()

    # file is not yet created
    except (FileNotFoundError, pd.errors.EmptyDataError):
        columns['no'] = '1'
        text = ','.join(columns.keys())
        print(columns.values())
        text += f'\n{",".join(columns.values())}'
        # create new file
        with open(file, 'w') as f:
            f.write(text)
            f.close()

    finally:
        print('model logged')

def log_model_train_scenario(
        scenario_num,
        scenario,
        epoch,
        drop_rate,
        learning_rate,
        weight_decay,
        vocabulary,
        n_heads,
        vocabulary_total,
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        time,
        file='alstm_model.csv',
    ):
    columns = {
        'no': 0,
        'scenario_num': str(scenario_num),
        'scenario': str(scenario),
        'epoch': str(epoch),
        'drop_rate': str(drop_rate),
        'learning_rate': str(learning_rate),
        'weight_decay': str(weight_decay),
        'vocabulary': str(vocabulary),
        'vocabulary_total': str(vocabulary_total),
        'n_heads': str(n_heads),
        'train_acc': str(train_acc),
        'test_acc': str(test_acc),
        'train_loss': str(train_loss),
        'test_loss': str(test_loss),
        'time': str(time)
    }

    try:
        df = pd.read_csv(file)
        last_index = df.shape[0]+1
        columns['no'] = str(last_index)
        
        text = ''
        text += f'\n{",".join(columns.values())}'

        with open(file, 'a') as f:
            f.write(text)
            f.close()

    # file is not yet created
    except (FileNotFoundError, pd.errors.EmptyDataError):
        columns['no'] = '1'
        text = ','.join(columns.keys())
        print(columns.values())
        text += f'\n{",".join(columns.values())}'
        # create new file
        with open(file, 'w') as f:
            f.write(text)
            f.close()

    finally:
        print('model logged')

def plot_model_history(
        history,
        title='model',
        save_path=None
        ):
    all_train_accs = history['train_accs']
    all_valid_accs = history['valid_accs']

    all_train_losses = history['train_losses']
    all_valid_losses = history['valid_losses']


    plt.figure(figsize=(16,6))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(all_valid_accs)), all_valid_accs, label='val_acc')
    plt.plot(range(len(all_train_accs)), all_train_accs, label='train_acc')
    plt.title('Grafik Akurasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(all_valid_losses)), all_valid_losses, label='val_loss')
    plt.plot(range(len(all_train_losses)), all_train_losses, label='train_loss')
    plt.title('Grafik Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def log_model_history(
        history,
        save_path,
):
    columns = ['epoch', 'train_acc', 'valid_acc', 'train_loss', 'valid_loss']

    text = ','.join(columns)

    # combine all value
    epochs = len(history['train_accs'])
    
    for epoch in range(epochs):
        train_acc = history['train_accs'][epoch]
        valid_acc = history['valid_accs'][epoch]
        train_loss = history['train_losses'][epoch]
        valid_loss = history['valid_losses'][epoch]
        text += f'\n{epoch+1},{train_acc},{valid_acc},{train_loss},{valid_loss},'
    
    with open(save_path, 'w') as f:
        f.write(text)
        f.close()

def log_report_and_conf_mat(
        title,
        report,
        confusion_matrix,
        label_names,
        save_path,
):
    plt.figure(figsize=(16,6))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Prediksi')
    plt.ylabel('Data Aktual')

    plt.subplot(1, 2, 2)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    pass