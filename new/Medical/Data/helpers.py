import pandas as pd


def combine_csvs(medical_csv, non_medical_csv, output_csv):
    # Load the CSV files
    medical_df = pd.read_csv(medical_csv)
    non_medical_df = pd.read_csv(non_medical_csv)

    # Add labels
    medical_df['label'] = 1
    non_medical_df['label'] = 0

    # Select the 'Extracted Text' column and filter out NaN values
    medical_df = medical_df[['Extracted Text', 'label']].dropna(subset=['Extracted Text'])
    non_medical_df = non_medical_df[['Extracted Text', 'label']].dropna(subset=['Extracted Text'])

    # Combine the dataframes
    combined_df = pd.concat([medical_df, non_medical_df])

    combined_df.rename(columns={'Extracted Text': 'text'}, inplace=True)

    # Save the combined dataframe to a new CSV
    combined_df.to_csv(output_csv, index=False)
