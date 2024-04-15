import json
import random
import os

def convert_to_jsonl(input_file, output_files, split_ratios=(0.8, 0.1, 0.1)):
    # Read JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Shuffle the data to ensure randomness in splits
    random.shuffle(data)

    # Calculate split indices
    num_examples = len(data)
    train_end = int(split_ratios[0] * num_examples)
    valid_end = train_end + int(split_ratios[1] * num_examples)

    # Split the data
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    # Function to save data to a jsonl file
    def save_to_jsonl(data_list, file_name):
        with open(file_name, 'w') as outfile:
            for entry in data_list:
                json.dump(entry, outfile)
                outfile.write('\n')  # Newline for next JSON object

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save splits to corresponding files in the same directory as the script
    save_to_jsonl(train_data, os.path.join(script_dir, output_files[0]))
    save_to_jsonl(valid_data, os.path.join(script_dir, output_files[1]))
    save_to_jsonl(test_data, os.path.join(script_dir, output_files[2]))

# Define input and output files
input_file = '/Users/thomaslim/rebel/data/biorel/triplets_linearized.json'
output_files = ['train.jsonl', 'valid.jsonl', 'test.jsonl']

# Run the conversion and splitting
convert_to_jsonl(input_file, output_files)

print("Files have been successfully created and split in the script's directory.")

