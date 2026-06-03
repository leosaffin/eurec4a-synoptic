"""
# GRIMM (aerosol) data
## Tidy and convert to netCDF

I've deleted the first two files because have a different set of intervals. I'm assuming
this was an initial setup thing.
- 202001221709.g6s
- 202001221851.g6s
- 202001221709.gtm
- 202001221851.gtm

The headers don't work with pandas. There is 12.5 and 17.5 but they are written as 12,5
and 17,5 so read as extra columns. There's always at least two empty columns at the end,
so this makes sense as the most likely reason.

The files .gtm is 1 minute samples and .g6s is 6-second samples. The .gtm files have
extra columns in the header (Berner1-4 and Anderson1-6). These are always empty though.
"""

from pathlib import Path

import numpy as np
import pandas as pd

actual_headers = [
    "DateTime",
    0.25,
    0.28,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.58,
    0.65,
    0.7,
    0.8,
    1,
    1.3,
    1.6,
    2,
    2.5,
    3,
    3.5,
    4,
    5,
    6.5,
    7.5,
    8.5,
    10,
    12.5,
    15,
    17.5,
    20,
    25,
    30,
    32,
    "Berner1",
    "Berner2",
    "Berner3",
    "Berner4",
    "Anderson1",
    "Andreson2",
    "Anderson3",
    "Anderson4",
    "Anderson5",
    "Anderson6",
]


def main():
    grimm_data = []
    for n, filename in enumerate(Path(".").glob("*.gtm")):
        df = pd.read_csv(filename, header=None, names=actual_headers, skiprows=2)

        if n == 0:
            columns = df.columns
        else:
            # Check for non-matching files
            # This shouldn't print anything now I have removed the first two files
            if len(columns) != len(df.columns) or (columns != df.columns).all():
                print(filename)
                print(df.columns)
                print()
            else:
                grimm_data.append(df)

    grimm_data = pd.concat(grimm_data)
    times = pd.to_datetime(grimm_data.DateTime, format="%d/%m/%Y %H:%M:%S")
    grimm_data["time"] = times
    grimm_data = (
        grimm_data.sort_values("time")
        .drop(columns="DateTime")
        .drop(columns=actual_headers[-10:])
    )

    # Changed NaNs to zeros. They were read as NaNs by pandas because they were empty columns
    grimm_data_xr = grimm_data.to_xarray().set_coords("time").drop_vars(["index"])
    for var in grimm_data_xr:
        grimm_data_xr[var][np.where(np.isnan(grimm_data_xr[var]))] = 0.0

    # Stack the columns (particle-size bins) as a second coordinated
    grimm_data_2d = grimm_data_xr.to_dataarray(
        dim="particle_size_bin_lower_boundary", name="particle_number_concentration"
    )
    grimm_data_2d.to_netcdf("grimm.nc")


if __name__ == "__main__":
    main()
