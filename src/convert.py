import pandas as pd
import xarray as xr

DATA_FILE_NAME = "../data/lipari-2.nc"

# Reading netcdf4 files as datasets
raw_dataset = xr.open_dataset(DATA_FILE_NAME)
# Converting datasets to dataframes
raw_dataframe = raw_dataset.to_dataframe()
# Reset indices
raw_dataframe.reset_index(inplace=True)

# Creating the dataframe for the "waves_parameters" table
waves_dataframe = pd.DataFrame()

mappings = {
    "lat": "latitude",
    "long": "longitude",
    "time": "time",
    "significant_wave_height": "swh",
    "energy_wave_period": "mwp",
    "mean_wave_direction": "mwd",
    "wind_100m_u": "u100", # Measured in m/s
    "wind_100m_v": "v100" # Measured in m/s
}

for name, raw_name in mappings.items():
    waves_dataframe[name] = raw_dataframe[raw_name]

waves_dataframe.index.names = ['index']
# waves_dataframe.to_csv("../data/wallis_north_wind_and_waves.csv")
waves_dataframe.to_csv("../data/lipari/wind_and_waves-2.csv")

print("Converted!")
