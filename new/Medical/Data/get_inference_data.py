import os
import pandas as pd
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def create_dataset(medical_dir, output_csv):
    data = []

    # Process medical PDFs
    for filename in os.listdir(medical_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(medical_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                data.append({"text": text, "label": 2})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

# Paths to your directories and output file
pdfs_dir = "../Data/Infer_Data"
output_csv_path = "../New_Data/test.csv"

# Create the dataset
create_dataset(pdfs_dir, output_csv_path)
