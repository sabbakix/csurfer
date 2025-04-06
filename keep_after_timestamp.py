import csv
import os


#keep data after timestamp:
st_timestamp = '1473155340.0'

# usage
data_directory = 'data/'  # Replace with your data directory path
input_csv = os.path.join(data_directory, 'btcusd_1-min_data.csv')  # Join directory and input file
output_csv = os.path.join(data_directory, 'btcusd_1-min_data_after'+st_timestamp+'.csv')  # Join directory and output file



def process_csv(input_file, output_file, st_timestamp):
    """
    Reads a large CSV file line by line, processes each line, and writes to a new CSV file.
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter=',')
        
        # Write the header row (if present)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        
        i=0
        keep=False
        # Process each row
        for row in reader:
            i=i+1
            if i%1000000==0:
                print(f"Processing row {i}")
                #break
            
            # Example operation: Convert all text in the row to uppercase
            # Replace all periods with commas in each cell of the row
            processed_row = []
            for cell in row:
                #print(cell)
                if cell == st_timestamp: 
                   keep=True
                   print("Found timestamp")

                # Check if the cell is a string and contains a period
                if keep:
                    processed_cell = cell
                    processed_row.append(processed_cell)
                    
            if keep:
                writer.writerow(processed_row)


process_csv(input_csv, output_csv, st_timestamp)