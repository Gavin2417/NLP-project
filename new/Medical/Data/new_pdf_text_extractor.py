import fitz  # PyMuPDF
import re
import os
import pandas as pd

# Define paths
input_directory = "../Data/pdfs/non_waste_burning"
output_csv = "../Data/cleaned_data/non_waste_burning.csv"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def remove_references(text):
    # Define headings that precede references
    reference_headings = [
        'References',
        'Bibliography',
        'Acknowledgments',
        'Acknowledgements',
        'Conflict of Interest',
        'Funding',
        'Abbreviations'
    ]

    # Create a regex pattern to search for these headings
    pattern = re.compile(r'\n({})\n'.format('|'.join(reference_headings)), re.IGNORECASE)

    # Split the text at the reference heading
    split_text = pattern.split(text)[0]
    return split_text.strip()


# Process all PDFs in the directory
data = []
for filename in os.listdir(input_directory):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(input_directory, filename)

        try:
            # Extract and clean the text
            text = extract_text_from_pdf(pdf_path)
            cleaned_text = remove_references(text)

            # Append to the data list
            data.append({"Filename": filename, "Extracted Text": cleaned_text})

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Data saved to {output_csv}")
