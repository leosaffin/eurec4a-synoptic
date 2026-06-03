import eurec4a
import numpy as np
import xarray as xr


sampling = "6h"
variables = ["VEL", "DIR", "T", "P", "R", "RH"]


def main():
    # Derive cloud fraction
    # https://howto.eurec4a.eu/bco_cloudradar.html
    cat = eurec4a.get_intake_catalog()
    ds = cat.barbados.bco.radar_reflectivity.to_dask()
    ds = ds.sel(time=slice("2020-01-20", "2020-02-20"))
    ds = ds.isel(range=ds.range < 5e3)

    cf = ds.Zf > -50
    cf = cf.resample(time=sampling).mean()

    cf.to_netcdf("cloud_fraction.nc")

    # Relevant surface fields
    # Following example provided by reviewer
    vars_to_save = {}
    wxt = cat.barbados.bco.meteorology.to_dask()
    for n, var in enumerate(variables):
        if var == "DIR":
            sindir = np.sin(np.deg2rad(wxt.DIR))
            sindir = sindir.resample(time=sampling).sum()
            cosdir = np.cos(np.deg2rad(wxt.DIR))
            cosdir = cosdir.resample(time=sampling).sum()
            y = np.rad2deg(np.atan2(sindir, cosdir))
        elif var == "R":
            y = wxt[var].resample(time=sampling).sum()
        else:
            y = wxt[var].resample(time=sampling).mean()

        vars_to_save[var] = y

    xr.Dataset(vars_to_save).to_netcdf("bco.nc")


if __name__ == "__main__":
    main()
