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

def create_dataset(medical_dir, non_medical_dir, output_csv):
    data = []

    # Process medical PDFs
    for filename in os.listdir(medical_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(medical_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                data.append({"text": text, "label": 1})

    # Process non-medical PDFs
    for filename in os.listdir(non_medical_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(non_medical_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                data.append({"text": text, "label": 0})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

# Paths to your directories and output file
# medical_pdfs_dir = "../waste_data/Yes"
# non_medical_pdfs_dir = "../waste_data/No"
# output_csv_path = "../New_Data/waste_train.csv"
medical_pdfs_dir = "../Data/Finetune_Data/Yes"
non_medical_pdfs_dir = "../Data/Finetune_Data/No"
output_csv_path = "../New_Data/train.csv"
# Create the dataset
create_dataset(medical_pdfs_dir, non_medical_pdfs_dir, output_csv_path)
