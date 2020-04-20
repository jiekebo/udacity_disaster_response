import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    
    # create a dataframe of the 36 individual category columns
    categories_split = categories['categories'].str.split(';', expand=True)
    categories_split.head()
    
    # select the first row of the categories dataframe
    row = categories_split.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = pd.Series(row).apply(lambda col_name : col_name[:-2])
    
    # rename the columns of `categories`
    categories_split.columns = category_colnames
    categories_split.head()
    
    
    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].str[-1]

        # convert column from string to numeric
        categories_split[column] = categories_split[column].astype(int)
    categories_split.head()
    
    # Add splitted categories to original index in order to join back with messages
    categories = pd.concat([categories.drop('categories', axis=1), categories_split], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    return messages.merge(categories, on='id')    


def clean_data(df):
    return df.drop_duplicates()


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()