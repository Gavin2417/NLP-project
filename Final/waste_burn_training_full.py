import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import os
import pandas as pd
import fitz
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',clean_up_tokenization_spaces=True)
accuracy_metric = evaluate.load("accuracy")

def chunk_text(text, tokenizer, max_length=512, stride=256):
    text = text.replace("\n", " ").strip()  # Clean up the text
    # print(text)
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )
    chunks = []
    # encoding["input_ids"] is a list of lists, one per chunk
    for input_ids in encoding["input_ids"]:
        text = tokenizer.decode(input_ids, skip_special_tokens=True).replace(" ", "").strip()
        chunks.append(text)
    return chunks

def expand_dataset(dataset):
    expanded_data = {"text": [], "label": []}
    for example in dataset:
        # Split the text into chunks
        chunks = chunk_text(example["text"], tokenizer, max_length=512)
        # For each chunk, store the chunk and the original label
        for chunk in chunks:
            expanded_data["text"].append(chunk)
            expanded_data["label"].append(example["label"])

    return Dataset.from_dict(expanded_data)

# Define a tokenization function for the chunks
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

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
def predict_label_for_pdf(pdf_path, model, tokenizer, device):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"No text extracted from {pdf_path}")
        return None

    # Split the full text into chunks that do not exceed 512 tokens
    chunks = chunk_text(text, tokenizer, max_length=512, stride=256)
    
    # Store logits from each chunk
    all_logits = []
    for chunk in chunks:
        # Tokenize each chunk (padding and truncation ensure fixed size input)
        inputs = tokenizer(chunk, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  
            all_logits.append(logits)
    
    # Aggregate predictions by averaging the logits across chunks.
    aggregated_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0, keepdim=True)  # shape [1, num_labels]
    predicted_label = torch.argmax(aggregated_logits, dim=1).item()
    return predicted_label

def process_pdfs_after_training(directory_path, output_csv, model, tokenizer, device):
    results = []
    
    # Iterate through all PDF files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            
            # Predict label for the PDF 
            predicted_label = predict_label_for_pdf(pdf_path, model, tokenizer, device)
            if predicted_label is None:
                continue  
            
            # Convert numeric prediction to string label
            label_map = {1: "YES", 0: "NO"}
            new_label = label_map.get(predicted_label, "UNKNOWN")
            results.append({"filename": filename, "predicted_label": new_label})
    
    # Save the results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on a limited number of samples per class.")
    parser.add_argument("--count", type=int, default=20, help="Number of 'YES' samples to use")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--folder", type=str, default="waste_40", help="Folder name for the dataset")
    args = parser.parse_args()
    
    waste_data_path = f"csv_data/{args.folder}.csv"
    waste_data = pd.read_csv(waste_data_path)
    # Filter and limit samples by label
    yes_samples = waste_data[waste_data["label"] == 1].head(args.count//2)
    no_samples = waste_data[waste_data["label"] == 0].head(args.count//2)
    
    print(f"Training on {len(yes_samples)} YES and {len(no_samples)} NO samples (total {len(waste_data)}).")

    # Combine and shuffle
    waste_data = pd.concat([yes_samples, no_samples]).sample(frac=1).reset_index(drop=True)
    print(f"Total samples after filtering: {len(waste_data)}")
    # Convert your DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(waste_data)
    # Use the custom function to expand the dataset
    dataset_chunked = expand_dataset(dataset)
    print(f"Total samples after chunking: {len(dataset_chunked)}")
    # Tokenize the chunked dataset
    dataset_tokenized = dataset_chunked.map(tokenize_function, batched=True)
    dataset_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load the model 
    waste_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    waste_model.load_state_dict(torch.load("medical_modle.pth", map_location=device, weights_only=True), strict=False)
    waste_model = waste_model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./temp",
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=2
    )

    # Initialize Trainer
    trainer = Trainer(
        model=waste_model,
        args=training_args,
        train_dataset=dataset_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Fine-tune the model on the chunked training data
    trainer.train()

    pdfs_dir = "pdf_data/test"
    # Create output directory if it doesn't exist
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists(f"result/{args.folder}"):
        os.makedirs(f"result/{args.folder}")
    output_csv_path = f"result/{args.folder}/test_predictions_{args.count}_{args.epochs}_half.csv"

    process_pdfs_after_training(
        directory_path=pdfs_dir,
        output_csv=output_csv_path,
        model=waste_model,       
        tokenizer=tokenizer,     
        device=device            
    )
