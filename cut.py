import csv
import os


#keep data after timestamp:
st_timestamp = '1650470340.0'
input_csv_file='btcusd_1-min_data.csv'

#get the base name of the file witout the csv extension
input_csv_file_base = os.path.splitext(input_csv_file)[0]  # Remove the .csv extension

# usage
data_directory = 'data/'  # Replace with your data directory path
input_csv = os.path.join(data_directory, input_csv_file)  # Join directory and input file
output_csv_before = os.path.join(data_directory, input_csv_file_base+'_'+st_timestamp+'_data_before.csv')  # Join directory and output file
output_csv_after = os.path.join(data_directory, input_csv_file_base+'_'+st_timestamp+'_data_after.csv')  # Join directory and output file



def process_csv(input_csv, output_csv_before, output_csv_after, st_timestamp):
    """
    Reads a large CSV file line by line, processes each line, and writes to a new CSV file.
    """



    print(f"Processing before "+st_timestamp+" ...")
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv_before, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter=',')
        
        # Write the header row (if present)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        
        i=0
        keep=True
        # Process each row
        for row in reader:
            i=i+1
            if i%500000==0:
                print(f"Processing row {i}")
                #break
            
            # Example operation: Convert all text in the row to uppercase
            # Replace all periods with commas in each cell of the row
            processed_row = []
            for cell in row:
                #print(cell)
                if cell == st_timestamp: 
                   keep=False
                   print("Found timestamp")

                # Check if the cell is a string and contains a period
                if keep:
                    processed_cell = cell
                    processed_row.append(processed_cell)
                    
            if keep:
                writer.writerow(processed_row)





    print(f"Processing after "+st_timestamp+" ...")
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv_after, mode='w', newline='', encoding='utf-8') as outfile:
        
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
            if i%500000==0:
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




process_csv(input_csv, output_csv_before, output_csv_after, st_timestamp)