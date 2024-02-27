import argparse
import json

# Create an argument parser
parser = argparse.ArgumentParser(description="Convert JSON metadata to a comma-separated string for ease of training. Example: python scripts/json_to_string.py -i metadata/histogram_lists/L1T.json -d L1T")
parser.add_argument('-i', '--input', type=str, help='Path to the input JSON file')
parser.add_argument('-d', '--detector', type=str, help='Enter detector (CSC, DT, L1T, etc.)')

# Parse the command-line arguments
args = parser.parse_args()

# Check if the input argument is provided
if args.input:
    input_path = args.input
else:
    print("Please provide the input JSON file using the -i flag.")
    exit()
if args.detector:
    detector = args.detector
else:
    print("Please provide the data set family (e.g. L1T, Muon) using the -d flag.")
    exit()

# Load the JSON data from the file
with open(input_path, 'r') as file:
    json_data = json.load(file)

# Extract the list from the JSON data
data_list = json_data[detector]
data_header = list(json_data.keys())[0]

# Convert the list into a comma-separated string
csv_string = ",".join([data_header + "/" + value for value in data_list])

# Print the comma-separated string
print(csv_string)
