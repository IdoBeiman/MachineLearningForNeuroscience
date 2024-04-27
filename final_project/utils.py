import pandas as pd

def extract_and_save_distinct_journals_with_impact_factors(journals_file_path, impact_factors_file_path, output_file):
    # Load the dataset with journal names and ensure they are distinct
    journals_data = pd.read_csv(journals_file_path).drop_duplicates()

    # Load the dataset with impact factors
    impact_factors_data = pd.read_csv(impact_factors_file_path)

    # Normalize the case for matching purposes
    journals_data['journal'] = journals_data['journal'].str.strip().str.lower()
    impact_factors_data['Journal Name'] = impact_factors_data['Journal Name'].str.strip().str.lower()

    # Merge the datasets on the journal names
    merged_data = journals_data.merge(impact_factors_data, left_on='journal', right_on='Journal Name', how='left')

    # Drop any duplicates that may result from merging
    merged_data.drop_duplicates(subset='journal', keep='first', inplace=True)

    # Replace NaN with 'Unknown' for journals without a matching impact factor
    merged_data['2021 JIF'] = merged_data['2021 JIF'].fillna('Unknown')

    # Select only the columns of interest and write the result to a text file
    sorted_merged_data = merged_data.sort_values(by='journal')
    sorted_merged_data[['journal', '2021 JIF']].to_csv(output_file, index=False, header=True)
    
    print(f"Distinct journal names with corresponding 2021 JIF have been written to {output_file}")

def convert_txt_to_csv(input_txt_file, output_csv_file):
    # Read the text file into a DataFrame
    df = pd.read_csv(input_txt_file, header=None, names=['Journal', 'Impact Factor'])
    
    # Sort the DataFrame alphabetically by journal name
    df_sorted = df.sort_values(by='Journal')
    
    # Write the sorted DataFrame to a CSV file
    df_sorted.to_csv(output_csv_file, index=False)

def extract_and_save_journals(file_path, output_file):
    # Load the dataset from the specified CSV file
    data = pd.read_csv(file_path)

    # Extract the distinct journals from the 'journal' column
    unique_journals = data['journal'].dropna().unique()

    # Sort the journals alphabetically
    unique_journals.sort()
    # Write the list of journals to a text file
    with open(output_file, 'w') as f:
        for journal in unique_journals:
            f.write(journal + '\n')