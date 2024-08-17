import csv
import random

def split_csv(input_filename, train_filename, test_filename, train_ratio=0.7):
    # Read the data from the input CSV file
    with open(input_filename, 'r') as infile:
        reader = csv.reader(infile)
        headers = next(reader)  # Get the headers
        rows = list(reader)  # Get the data rows
    
    # Shuffle the rows to ensure random split
    random.shuffle(rows)
    
    # Calculate the split point
    split_point = int(len(rows) * train_ratio)
    
    # Split the data into training and testing sets
    train_rows = rows[:split_point]
    test_rows = rows[split_point:]
    
    # Write the training set to the train file
    with open(train_filename, 'w', newline='') as train_file:
        writer = csv.writer(train_file)
        writer.writerow(headers)  # Write the headers
        writer.writerows(train_rows)  # Write the training data
    
    # Write the testing set to the test file
    with open(test_filename, 'w', newline='') as test_file:
        writer = csv.writer(test_file)
        writer.writerow(headers)  # Write the headers
        writer.writerows(test_rows)  # Write the testing data

# Example usage
split_csv('LGBM1.csv', 'LGBM1_train.csv', 'LGBM1_test.csv')