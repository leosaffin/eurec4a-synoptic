import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import (
    snapshot_times,
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

    ds = xr.open_dataset(
        "daily_m_y-u_y-omega_y-global_allplev-2020-01-20_2020-02-20.nc"
    )
    ds = ds.sel(time=snapshot_times, level=500)
    ds = roll_lons(ds)

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

    for i, time in enumerate(pd.to_datetime(snapshot_times), start=1):
        print(time)
        mean_eureca_1d = ds.sel(time=time)

        ax = plt.axes(axes[str(i)])
        im = ax.pcolormesh(
            mean_eureca_1d.longitude[::4],
            mean_eureca_1d.latitude[::4],
            mean_eureca_1d.m_y[::4, ::4],
            transform=transform,
            vmin=-vmax,
            vmax=vmax,
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

        ax.set_title(time.strftime("%Y-%m-%d"))

    plt.colorbar(
        im,
        cax=axes["c"],
        orientation="horizontal",
        label="kg m$^2$ s$^{-1}$",
        extend="both",
    )
    label_axes([axes[str(n)] for n in range(1, 4 + 1)])
    plt.savefig("hadley_snapshots.pdf")
    plt.show()


if __name__ == "__main__":
    main()
