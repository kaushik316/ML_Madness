# ML_Madness
A machine learning model to (attempt to) predict the NCAA Basketball March Madness tournament. 
Blog post on this project [here](http://kaushik316-blog.logdown.com/posts/1675209-march-madness-betrayed-by-a-machine).

### Usage
Once you have all the files in the data folder downloaded, clean the files using the code in the notebooks folder. Order of execution is:
1. **calc_stats.ipynb**: Converts some of the stats provided to percentages and drops unnnecessary columns.
2. **scrape_espn.ipynb**: Obtains metrics from ESPN such as BPI and strength of schedule for each team since 2007.
3. **final_dataprep.ipynb**: Combines ESPN dataframe with tournament dataframe and performs final data processing.

Once these files are run, the processed data will be written to a file in the proc_data folder. When the `CombinedResults.csv` and `FinalStats.csv` files are generated, run `python model.py` for predictions
