import json
import random
import os

def convert_to_jsonl(input_file):
    # Read JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Function to save data to a jsonl file
    def save_to_jsonl(data, file_name):
        with open(file_name, 'w') as outfile:
            for entry in data:
                json.dump(entry, outfile)
                outfile.write('\n')  # Newline for next JSON object

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save splits to corresponding files in the same directory as the script
    save_to_jsonl(data, os.path.join(script_dir, "chemprot_sample.jsonl"))

# Define input and output files
input_file = '/Users/thomaslim/rebel/data/chemprot/chemprot_sample_linearized.json'
# Run the conversion and splitting
convert_to_jsonl(input_file)


