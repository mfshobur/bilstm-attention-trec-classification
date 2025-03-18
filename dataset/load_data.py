from datasets import load_dataset

# Load the 'trec' dataset
dataset = load_dataset("trec")

# Check the available splits
print(dataset)

# Access the train and test splits
train_data = dataset['train']
test_data = dataset['test']

from datasets import load_dataset

# Load the TREC dataset
dataset = load_dataset("trec")

# Save the train and test splits to CSV files
dataset['train'].to_csv("trec_train.csv", index=False)
dataset['test'].to_csv("trec_test.csv", index=False)

print("Train and test data saved to 'trec_train.csv' and 'trec_test.csv'")
