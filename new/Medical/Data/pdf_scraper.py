import fitz  # PyMuPDF
import csv
import os
import pandas as pd
from pymupdf import FileDataError


def extract_text_blocks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_blocks = []

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip() 
            if text:  
                text_blocks.append(text)

    return text_blocks

def get_body(string_list, start_string, end_string):
    
    lower_list = [s.lower() for s in string_list]
    start_string_lower = start_string.lower()
    end_string_lower = end_string.lower()

    if start_string_lower not in lower_list or end_string_lower not in lower_list:
        return []

    start_index = lower_list.index(start_string_lower)
    end_index = lower_list.index(end_string_lower)

    if start_index >= end_index:
        return []

    return string_list[start_index + 1:end_index]

# # Example usage
# pdf_path = "./pdfs/non_medical/2401.10568v2.pdf"
# text_blocks = extract_text_blocks_from_pdf(pdf_path)
# #TODO find way to remove headings
# new_text_block = get_body(text_blocks,"Abstract","References")
# string_body = ' '.join(new_text_block)
# print(string_body)

# # # Print extracted text blocks
# for block in text_blocks:
#     print(block)

def process_pdfs_in_directory(directory, start_string, end_string, output_csv):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            try:
                text_blocks = extract_text_blocks_from_pdf(pdf_path)
            except FileDataError:
                print("passed")
                continue
            new_text_block = get_body(text_blocks, start_string, end_string)
            string_body = ' '.join(new_text_block)
            # Append the data with the filename and extracted body text
            data.append({"Filename": filename, "Extracted Text": string_body})

    # Create a DataFrame and write it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=True, encoding='utf-8')

directory = "../Data/pdfs/waste_burning"
start_string = "Abstract"
end_string = "References"
output_csv = "../Data/cleaned_data/waste_burning.csv"

process_pdfs_in_directory(directory, start_string, end_string, output_csv)