import csv
import os

def process_csv(input_file, output_file):
    """
    Reads a large CSV file line by line, processes each line, and writes to a new CSV file.
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter=';')
        
        # Write the header row (if present)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        
        i=0
        # Process each row
        for row in reader:
            i=i+1
            if i%500000==0:
                print(f"Processing row {i}")
                #break
            # Example operation: Convert all text in the row to uppercase
            processed_row = [ cell.replace(".",",") for cell in row]
            writer.writerow(processed_row)

# Example usage
data_directory = 'data/'  # Replace with your data directory path


input_csv = os.path.join(data_directory, 'btcusd_1-min_data.csv')  # Join directory and input file
output_csv = os.path.join(data_directory, 'btcusd_1-min_data_v2.csv')  # Join directory and output file


process_csv(input_csv, output_csv)