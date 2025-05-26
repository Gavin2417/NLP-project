import os
import pandas as pd
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text().strip()
        return full_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def create_dataset(folder_names, output_csv = None):
    data = []
    original_folder = "../Final/pdf_data/"
    for i in range(len(folder_names)):
        yes_pdfs_dir = os.path.join(original_folder, folder_names[i], "Yes")
        no_pdfs_dir = os.path.join(original_folder, folder_names[i], "No")

        # Process yes PDFs (Yes can be medical, waste, etc.)
        for filename in os.listdir(yes_pdfs_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(yes_pdfs_dir, filename)
                text = extract_text_from_pdf(pdf_path)
                if text:
                    data.append({"text": text, "label": 1})

        # Process no PDFs
        for filename in os.listdir(no_pdfs_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(no_pdfs_dir, filename)
                text = extract_text_from_pdf(pdf_path)
                if text:
                    data.append({"text": text, "label": 0})

    print(f"Processed {len(data)} PDFs in total.")
    # Convert to DataFrame
    df = pd.DataFrame(data)

    file_name = f"waste_{len(data)}.csv" 
    if output_csv is None:
        output_csv = os.path.join("../Final/csv_data", file_name)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

if __name__ == "__main__":

    # Paths to your directories and output file Y
    # file_names = ['waste_12', 'waste_37', 'waste_40']
    file_names = ['waste_12','waste_37',  'waste_40']
    # output_csv_name  = 'waste_37'
    create_dataset(file_names)
