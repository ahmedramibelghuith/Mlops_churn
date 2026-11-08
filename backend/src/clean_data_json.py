"""JSON and CSV requests deliberately share the same preprocessing."""
from .clean_data_csv import clean_data

clean_data_json = clean_data
