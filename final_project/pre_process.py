import csv
import pandas as pd
from pandas import DataFrame
from sklearn.discriminant_analysis import StandardScaler

HOT_ENCODE_COLS = ['journal']
ENCODING_PREFIX = 'journal_encoded'
def merge_impact_factors_with_dataset(original_data:DataFrame, impact_factors_path, output_dataset_path):
    # Load the impact factors dataset
    impact_factors = pd.read_csv(impact_factors_path)
    
    # Normalize the case for matching purposes
    original_data['journal'] = original_data['journal'].str.strip().str.lower()
    impact_factors['Journal'] = impact_factors['Journal'].str.strip().str.lower()
    
    # Merge the datasets on the journal names, ensuring we do a left join to keep all records from the original dataset
    merged_data = original_data.merge(impact_factors, left_on='journal', right_on='Journal', how='left')
    merged_data.drop(columns=['Journal'], inplace=True)

    # Scale the Impact Factor column
    scaler = StandardScaler()
    merged_data['Impact Factor'] = scaler.fit_transform(merged_data[['Impact Factor']])  # as impact factor is on a different scale we need to scale it
    return merged_data

def create_new_csv_with_first_5_rows(existing_csv_file, new_csv_file):
    # Open the existing CSV file for reading
    with open(existing_csv_file, mode='r', newline='') as existing_file:
        # Create a CSV reader
        csv_reader = csv.reader(existing_file)
        
        # Read the first 5 rows from the existing CSV file
        rows = [next(csv_reader) for _ in range(5)]
        
    # Open the new CSV file for writing
    with open(new_csv_file, mode='w', newline='') as new_file:
        # Create a CSV writer
        csv_writer = csv.writer(new_file)
        
        # Write the first 5 rows to the new CSV file
        csv_writer.writerows(rows)
        
    print("New CSV file with the first 5 rows created successfully.")

def hot_encode_column(data):
    data = pd.get_dummies(data, columns=HOT_ENCODE_COLS, drop_first=True, prefix=ENCODING_PREFIX, prefix_sep='_')
    return data

def pre_process_dataset(csv_file):
    data = pd.read_csv(csv_file)
    data = merge_impact_factors_with_dataset(data, "data/impact_factors.csv", "data/merged_data_234.csv")
    data = hot_encode_column(data)
    missing_citation_count = data['citation_count'].isnull()

    # Print indices of rows with missing "citation_count" values
    if missing_citation_count.any():
        print("Rows with missing 'citation_count':")
        print(missing_citation_count[missing_citation_count].index.tolist())
        
        # Delete rows where 'citation_count' is missing
        data_cleaned = data.dropna(subset=['citation_count'])
        print("Rows with missing 'citation_count' have been deleted.")
    else:
        print("No missing values in 'citation_count' column.")
        data_cleaned = data

    missing_data = data_cleaned.isnull()
    if missing_data.any().any():
        print("Remaining rows and columns with missing values after deletion:")
        for index, row in missing_data.iterrows():
            if row.any():
                missing_columns = row[row].index.tolist()
                print(f"Row {index} has missing values in columns: {missing_columns}")
    else:
        print("No other missing values found in the dataset after deleting rows with missing 'citation_count'.")
    data_cleaned = data_cleaned.reset_index(drop=True)
    data_cleaned.to_csv("data/filtered_df.csv")
    return data_cleaned


def convert_tsv_to_csv(tsv_file, csv_file):
    # Open the TSV file for reading
    with open(tsv_file, mode='r', newline='') as tsv_file:
        # Create a CSV reader that splits on tabs
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        
        # Open the CSV file for writing
        with open(csv_file, mode='w', newline='') as csv_file:
            # Create a CSV writer that uses commas as delimiters
            csv_writer = csv.writer(csv_file, delimiter=',')
            
            # Read each row from the TSV and write to the CSV file
            for row in tsv_reader:
                csv_writer.writerow(row)

    print("Conversion from TSV to CSV completed successfully.")

# create_new_csv_with_first_5_rows("data/data-neurosynth_version-7_features_with_citations.csv", "data/first_5_rows.csv")
#filtered_df = check_for_null_values("data/data-neurosynth_version-7_features_with_citations.csv")