import csv
import time
import os

# --- Configuration ---
# NOTE: Replace 'product_data.txt' with the actual path to your 144-page document.
INPUT_FILEPATH = '/home/dfowler/csv/marapr.txt'
# Changed output to .txt for plain text output
OUTPUT_FILEPATH = '/home/dfowler/documents/maraprcopy.txt'

def process_large_file(input_filepath, output_filepath):
    """
    Reads a large text file line-by-line to efficiently extract product item numbers
    starting with 'AG' and the preceding one or two lines (the description).

    This approach is efficient for large files (batch processing concept) because
    it processes the file line by line without loading the entire content into memory.
    """
    
    print(f"Starting extraction process from: {input_filepath}")
    
    # Initialize buffers to hold the last one and two non-empty lines
    line_minus_one = ""  # The line immediately before the item number
    line_minus_two = ""  # The line two lines before the item number
    
    line_count = 0
    start_time = time.time()

    # The processing is wrapped in a file writer for structured output
    try:
        with open(input_filepath, 'r', encoding='latin-1') as infile:
            # Open output file for writing
            with open(output_filepath, 'w', encoding='latin-1') as outfile:
                
                # Write a simple header line for context
                outfile.write("Item Number\tProduct Description\n")

                for line in infile:
                    line_count += 1
                    current_line = line.strip()

                    # 1. Check for the Target Item Number
                    if current_line.startswith('AG') and len(current_line) > 2:
                        item_number = current_line
                        
                        # Concatenate the buffered lines to form the description.
                        # We use ' ' only if line_minus-two actually has content.
                        if line_minus_two:
                            description = f"{line_minus_two} {line_minus_one}"
                        else:
                            description = line_minus_one
                        
                        # Only include results if a description (line_minus_one) was found
                        if description:
                            # Write the result to the text file, separated by a tab (\t)
                            outfile.write(f"{item_number}\t{description}\n")
                            
                        # Reset the buffers immediately after a successful extraction
                        line_minus_one = ""
                        line_minus_two = ""
                        
                    # 2. Update the Line Buffer (only for non-empty lines)
                    elif current_line:
                        # Move the lines down the buffer stack
                        line_minus_two = line_minus_one
                        line_minus_one = current_line
                    
                    # 3. Timed/Batch Progress Report (for large files)
                    if line_count % 1000 == 0:
                        print(f"Processed {line_count} lines... Keep going...")
                        
    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{input_filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        return

    end_time = time.time()
    
    print("\n--- Processing Complete ---")
    print(f"Total lines read: {line_count}")
    print(f"Results saved to: {os.path.abspath(output_filepath)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


# --- Example Usage (simulating the input file content) ---
def create_sample_file(filepath):
    """Creates a small sample file to make the script runnable."""
    sample_content = """
    This is an unrelated header.
    
    This is the first sentence of a description.
    And this is the second, explaining the durability.
    AG903421
    
    Item AG887 has a description on the line before.
    AG887123
    
    Just a note without an AG item number.
    
    Another product line. This description is only one line long.
    AG5500B
    
    AG200 is here but has no description above it.
    AG200000
    
    End of file.
    """
    print(f"Creating sample file at '{filepath}'...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_content)


if __name__ == '__main__':
    
    process_large_file(INPUT_FILEPATH, OUTPUT_FILEPATH)
