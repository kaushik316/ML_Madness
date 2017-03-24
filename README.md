# ML_Madness
A machine learning model to (attempt to) predict the NCAA Basketball March Madness tournament

### Usage
Once you have all the files in the data folder downloaded, clean the files using the code in the notebooks folder. Order of execution is:
1. calc_stats.ipynb: Converts some of the stats provided to percentages and drops unnnecessary columns.
2. scrape_espn.ipynb: Obtains metrics from ESPN such as BPI and strength of schedule for each team since 2007.
3. final_dataprep.ipynb: Combines ESPN dataframe with tournament dataframe and performs final data processing.

Once these files are run, the processed data will be written to a file in the proc_data folder. When these files are generated, run model.py for predictions
