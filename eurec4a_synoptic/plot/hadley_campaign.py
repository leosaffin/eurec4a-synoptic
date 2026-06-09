import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from . import (
    mslp_overlay,
    add_halo_circle,
    projection,
    transform,
    extent,
    roll_lons,
    label_axes,
)
from .hadley_cross_sections import plot_single_hadley


def main():
    vmax = 0.01

    ds = xr.open_dataset("era5_eurec4a.nc")
    mean_mslp = ds.msl.mean(dim="time").compute() / 100

    # mean local Hadley circulation map
    mean_eureca = xr.open_dataset("mean_m_y_eureca-timepsan.nc")
    mean_eureca = roll_lons(mean_eureca)

    # Anomaly from long term mean
    ds_l30y = xr.open_dataset("mean_500hPa_m_y_1990-2023.nc")
    data = roll_lons(ds_l30y)

    # Mean cross section
    topo = xr.load_dataarray("mean_sfp_1979-2023.nc")

    fig, axes = plt.subplot_mosaic(
        """
        111111222222
        333333xxbyyy
        """,
        figsize=(8, 6),
        per_subplot_kw={str(n): dict(projection=projection) for n in [1, 2]},
    )
    axes["x"].set_axis_off()
    axes["y"].set_axis_off()

    for ax, da in [
        (axes["1"], mean_eureca.m_y.sel(level=500)),
        (axes["2"], mean_eureca.m_y.sel(level=500) - data.m_y),
    ]:
        im = ax.pcolormesh(
            da.longitude,
            da.latitude,
            da,
            transform=transform,
            vmin=-vmax,
            vmax=vmax,
            cmap="seismic",
        )

        add_halo_circle(ax, transform=transform)
        ax.set_extent(extent, crs=transform)
        mslp_overlay(ax, mean_mslp)
        ax.coastlines(resolution="10m")

    axes["1"].gridlines(draw_labels=["left", "bottom"])
    axes["2"].gridlines(draw_labels=["bottom"])

    axes["1"].plot()

    im = plot_single_hadley(
        mean_eureca,
        axes["3"],
        lon1=-60,
        lon2=-40,
        skip=5,
        vskip=1,
        cmap="seismic",
        vmax=vmax,
        topo=topo,
        add_key=True,
    )
    axes["3"].set_xlim(-40, 40)
    plt.colorbar(im, cax=axes["b"], label="kg m$^2$ s$^{-1}$", extend="both")

    axes["3"].set_ylabel("Pressure (hPa)")
    axes["1"].set_title("EUREC$^4$A Mean\n(20th Jan - 20th Feb)")
    axes["2"].set_title("EUREC$^4$A Anomaly\n(vs 1990-2023)")
    axes["3"].set_title("EUREC$^4$A Mean")
    label_axes([axes[str(n)] for n in range(1, 3 + 1)])

    plt.savefig("hadley_campaign.pdf")
    plt.show()


if __name__ == "__main__":
    main()
