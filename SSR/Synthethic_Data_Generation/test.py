import jsonlines

def extract_last_outputs(jsonl_path, output_txt_path):
    """
    Extracts the last output from each sample in a JSONL file and saves it to a text file.

    Parameters:
        jsonl_path (str): Path to the input JSONL file.
        output_txt_path (str): Path to the output text file where extracted outputs will be saved.
    """
    last_outputs = []

    with jsonlines.open(jsonl_path, mode="r") as reader:
        for obj in reader:
            # Check if 'output' key exists
            if 'output' in obj:
                output = obj['output']
                # Split the output into lines and get the last one
                last_line = output.strip().split('\n')[-1]
                last_outputs.append(last_line)

    # Save the extracted last outputs to a text file
    with open(output_txt_path, mode="w", encoding="utf-8") as writer:
        for output in last_outputs:
            writer.write(output + "\n")

# Example usage
jsonl_path = "qa.train.final_sampled.jsonl"  # Replace with your JSONL file path
output_txt_path = "qa_final.txt"  # Replace with your desired output file path

extract_last_outputs(jsonl_path, output_txt_path)
print(f"Last outputs have been extracted to {output_txt_path}.")
