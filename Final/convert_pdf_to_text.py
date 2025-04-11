import os
import pandas as pd
import fitz  # PyMuPDF
import re
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text().strip()
        # pattern = r'(?mi)^\s*(References|Bibliography|Literature\s+Cited)\s*$'
    
        # # Search for the pattern in the text
        # match = re.search(pattern, full_text)
        # if match:
        #     # Remove everything from the header onwards
        #     full_text = full_text[:match.start()]
        
        return full_text
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
medical_pdfs_dir = "../NLP/pdf_data/waste_40/Yes"
non_medical_pdfs_dir = "../NLP/pdf_data/waste_40/No"
output_csv_path = "../NLP/csv_data/waste_40.csv"
# Create the dataset
create_dataset(medical_pdfs_dir, non_medical_pdfs_dir, output_csv_path)
