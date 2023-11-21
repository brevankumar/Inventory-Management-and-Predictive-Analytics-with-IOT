import os
from src.mlProject import logger
import pandas as pd
from src.mlProject.entity.config_entity import DataValidationConfig
from datetime import datetime
from src.mlProject.utils.common import convert_to_datetime,convert_timestamp_to_hourly
from sklearn.model_selection import train_test_split

                                    

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def process_and_save_csv(self):

        try:

            # Create the output folder if it doesn't exist
            os.makedirs(self.config.output_folder, exist_ok=True)

            # List all CSV files in the input folder
            csv_files = [file for file in os.listdir(self.config.input_folder) if file.endswith('.csv')]

            # Process each CSV file
            for csv_file in csv_files:
                # Construct the full path for input and output files
                input_path = os.path.join(self.config.input_folder, csv_file)
                output_path = os.path.join(self.config.output_folder, f"Dropped_validated_{csv_file}")

                # Load the CSV file into a DataFrame
                df = pd.read_csv(input_path)
                
                # Drop different columns in each CSV file
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])

                if 'id' in df.columns:
                    df = df.drop(columns=['id'])

                if 'transaction_id' in df.columns:
                    df = df.drop(columns=['transaction_id'])

            # Save the modified DataFrame to a new CSV file
            df.to_csv(output_path, index=False)
        
        except Exception as e:
            raise e
        


    def merge_and_save(self):
        try:
            # Read input files into dataframes
            df1 = pd.read_csv(self.config.input_file1)
            df2 = pd.read_csv(self.config.input_file2)
            df3 = pd.read_csv(self.config.input_file3)

            sales_df = convert_to_datetime(df1, 'timestamp')
            stock_df = convert_to_datetime(df2, 'timestamp')
            temp_df = convert_to_datetime(df3, 'timestamp')

            sales = convert_timestamp_to_hourly(sales_df, 'timestamp')
            sales_agg = sales.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()

            stock = convert_timestamp_to_hourly(stock_df, 'timestamp')
            stock_agg = stock.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()

            temp = convert_timestamp_to_hourly(temp_df, 'timestamp')
            temp_agg = temp.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()

            merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
            merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')

            merged_df['quantity'] = merged_df['quantity'].fillna(0)

            product_categories = sales_df[['product_id', 'category']]
            product_categories = product_categories.drop_duplicates()

            product_price = sales_df[['product_id', 'unit_price']]
            product_price = product_price.drop_duplicates()

            merged_df = merged_df.merge(product_categories, on="product_id", how="left")
            merged_df = merged_df.merge(product_price, on="product_id", how="left")

            merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
            merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
            merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
            merged_df.drop(columns=['timestamp','product_id'], inplace=True)


            # Create the output directory if it doesn't exist
            os.makedirs(self.config.output_directory, exist_ok=True)

            # Define the output file path
            output_file_path = os.path.join(self.config.output_directory, 'merged_output.csv')

            # Save the merged dataframe to the specified directory
            merged_df.to_csv(output_file_path, index=False)

        except Exception as e:
            raise e



    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.merged_output)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e

    def train_test_spliting(self):
        try:

            data = pd.read_csv(self.config.merged_output)

            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(data)

            train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

            logger.info("Splited data into training and test sets")
            logger.info(train.shape)
            logger.info(test.shape)

        except Exception as e:
            raise e