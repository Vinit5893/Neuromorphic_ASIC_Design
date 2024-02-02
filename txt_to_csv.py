import pandas as pd

# Assuming 'input.txt' is your text file and 'output.csv' is the desired CSV file
input_file = '100annotations.txt'
output_file = '100output.csv'

# Read the text file into a DataFrame using pandas
df = pd.read_csv(input_file, delimiter=r'\s+', engine='python')

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)