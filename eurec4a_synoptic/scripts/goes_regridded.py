import datetime

import cartopy.crs as ccrs
from goes2go import GOES
import numpy as np
from scipy.interpolate import griddata
import xarray as xr


def main():
    time = datetime.datetime(2020, 1, 20, 18)
    end_time = datetime.datetime(2020, 2, 20, 18)
    dt = datetime.timedelta(days=1)

    # target_lon = np.arange(-75, -20 + 0.1, 0.02)
    target_lon = np.arange(-100, 0 + 0.01, 0.1)
    # target_lat = np.arange(7.5, 25 + 0.1, 0.02)
    target_lat = np.arange(0, 45 + 0.01, 0.1)

    g16 = GOES(satellite=16, product="ABI-L2-MCMIP", domain="F")
    while time <= end_time:
        print(time)
        ds = g16.nearesttime(time, return_as="xarray")[
            "goes_imager_projection", "CMI_C01", "CMI_C02", "CMI_C03"
        ]
        print(ds)
        rgb = goes_regridded(
            ds,
            target_lon,
            target_lat,
            ["CMI_C02", "CMI_C03", "CMI_C01"],
            coarse_factor=1,
        )
        rgb.to_netcdf(f"goes_rgb_era5_grid_{time.strftime('%Y%m%dT%H%M')}.nc")

        time += dt
        stop


def goes_regridded(data_goes, target_lons, target_lats, variables, coarse_factor=1):
    p = data_goes["goes_imager_projection"]

    # The projection x and y coordinates equals the scanning angle (in radians)
    # multiplied by the satellite height See details here:
    # https://proj4.org/operations/projections/geos.html?highlight=geostationary
    x = data_goes.x.data[::coarse_factor] * p.perspective_point_height
    y = data_goes.y.data[::coarse_factor] * p.perspective_point_height
    X, Y = np.meshgrid(x, y)

    globe = ccrs.Globe(
        semimajor_axis=p.semi_major_axis,
        semiminor_axis=p.semi_minor_axis,
        inverse_flattening=p.inverse_flattening,
    )
    geos = ccrs.Geostationary(
        central_longitude=p.longitude_of_projection_origin,
        satellite_height=p.perspective_point_height,
        sweep_axis=p.sweep_angle_axis,
        globe=globe,
    )

    a = ccrs.PlateCarree().transform_points(geos, X, Y)
    lons, lats, _ = a[:, :, 0], a[:, :, 1], a[:, :, 2]

    print(lons.min(), lons.max())

    # Flatten and remove NaNs. The NaNs aren't in a square grid so can't be dropped on
    # loading
    outside_lons = (lons.flatten() < target_lons.min()) | (
        lons.flatten() > target_lons.max()
    )
    outside_lats = (lats.flatten() < target_lats.min()) | (
        lats.flatten() > target_lats.max()
    )
    mask = np.isnan(lons.flatten()) | outside_lons | outside_lats
    x = lons.flatten()[~mask]
    y = lats.flatten()[~mask]

    # Make sure to loop over variables as a list if only one is requested
    if isinstance(variables, str):
        variables = [variables]

    # Interpolate the satellite data to a regular grid
    target_lons_2d, target_lats_2d = np.meshgrid(target_lons, target_lats)
    goes_data_grid = xr.Dataset(
        coords=dict(longitude=target_lons, latitude=target_lats)
    )
    for variable in variables:
        print(variable)
        data = (
            data_goes[variable].data[::coarse_factor, ::coarse_factor].flatten()[~mask]
        )

        print(data.shape, x.shape, y.shape)
        variable_on_grid = griddata((x, y), data, (target_lons_2d, target_lats_2d))

        goes_data_grid[variable] = (["latitude", "longitude"], variable_on_grid)

    return goes_data_grid


if __name__ == "__main__":
    main()
