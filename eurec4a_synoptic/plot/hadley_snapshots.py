import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import (
    mslp_overlay,
    add_halo_circle,
    projection,
    transform,
    extent,
    label_axes,
    roll_lons,
)


def main():
    # Four maps
    vmax = 0.1
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 41)

    ds = xr.open_dataset("m_y_eureca.nc").sel(pressure_level=500)
    ds = roll_lons(ds).__xarray_dataarray_variable__

    mslp = xr.open_dataset("era5_eurec4a.nc").msl / 100

    fig, axes = plt.subplot_mosaic(
        """
        111222
        xxxxxx
        333444
        yyyyyy
        zccccw
        """,
        figsize=(8, 6),
        per_subplot_kw={str(n): dict(projection=projection) for n in range(1, 4 + 1)},
        height_ratios=[10, 1, 10, 1, 1],
    )
    for ax in ["w", "x", "y", "z"]:
        axes[ax].set_axis_off()

    for i, time in enumerate(pd.to_datetime(ds.valid_time.values), start=1):
        print(time)
        mean_eureca_1d = ds.sel(valid_time=time)

        print(mean_eureca_1d.values.min(), mean_eureca_1d.values.max())

        ax = plt.axes(axes[str(i)])

        im = ax.contourf(
            mean_eureca_1d.longitude,
            mean_eureca_1d.latitude,
            mean_eureca_1d,
            transform=transform,
            levels=levels,
            cmap="seismic",
        )

        add_halo_circle(ax, transform=transform)
        ax.coastlines(resolution="10m")

        if i == 1:
            ax.gridlines(draw_labels=["left"])
        elif i == 2:
            ax.gridlines()
        elif i == 3:
            ax.gridlines(draw_labels=["left", "bottom"])
        else:
            ax.gridlines(draw_labels=["bottom"])

        ax.set_extent(extent, crs=transform)

        mslp_overlay(ax, mslp.sel(time=time))

        ax.set_title(time.strftime("%Y-%m-%d %H:%M UTC"))

    plt.colorbar(im, cax=axes["c"], orientation="horizontal", label="kg m$^2$ s$^{-1}$")
    label_axes([axes[str(n)] for n in range(1, 4 + 1)])
    plt.savefig("hadley_snapshots.pdf")


if __name__ == "__main__":
    main()
