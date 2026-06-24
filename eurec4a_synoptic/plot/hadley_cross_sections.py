import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import snapshot_times, label_axes, roll_lons


def main():
    topo = xr.load_dataarray("mean_sfp_1979-2023.nc")
    ds = xr.open_dataset(
        "daily_m_y-u_y-omega_y-global_allplev-2020-01-20_2020-02-20.nc"
    ).sel(time=snapshot_times)
    ds = roll_lons(ds)

    fig, axes = plt.subplot_mosaic(
        """
        1111-2222
        1111-2222
        1111-2222
        1111-2222
        wwww-xxxx
        3333-4444
        3333-4444
        3333-4444
        3333-4444
        yyyy-zzzz
        acccccccb
        """,
        figsize=(8, 6),
    )
    for ax in ["-", "a", "b", "x", "y", "z", "w"]:
        axes[ax].set_axis_off()

    for n, time in enumerate(pd.to_datetime(snapshot_times), start=1):
        ax = axes[str(n)]
        im = plot_single_hadley(
            ds.sel(time=time),
            ax,
            -60,
            -40,
            skip=10,
            vskip=2,
            cmap="seismic",
            vmax=0.05,
            topo=topo,
            scale=1e3,
            add_key=n == 1,
        )
        ax.set_title(time.strftime("%Y-%m-%d"))
        ax.set_xlim(-40, 40)

        if n % 2 == 0:
            ax.set_yticklabels([])
        if n < 3:
            ax.set_xticklabels([])

    plt.colorbar(
        im,
        cax=axes["c"],
        orientation="horizontal",
        label="kg m$^2$ s$^{-1}$",
        extend="both",
    )
    fig.text(0.05, 0.55, "Pressure (hPa)", rotation="vertical", va="center")
    label_axes([axes[str(n)] for n in range(1, 4 + 1)])

    plt.savefig("hadley_cross_sections.pdf")
    plt.show()


def cut_topo(ds, surface_pressure):
    """
    Masks out values where the pressure level is above the surface pressure
    (i.e., underneath the topography).

    Parameters:
    ds (xr.Dataset): Dataset containing variables on pressure levels
        (dimensions: level, latitude, longitude).
    surface_pressure (xr.DataArray): Mean surface pressure (hPa) with dimensions
        (latitude, longitude).
    pressure_levels (list): List of pressure levels in hPa (e.g., [850, 700, 500]).

    Returns:
    xr.Dataset: Masked dataset where values below topography are set to NaN.
    """
    ds_masked = ds.copy()

    pressure_levels = ds.level

    for p_level in pressure_levels:
        mask = (
            surface_pressure < p_level
        )  # True where p_surf is less than pressure level (masked out)
        ds_masked.loc[dict(level=p_level)] = ds.loc[dict(level=p_level)].where(~mask)

    return ds_masked


def lambda_average(lon1, lon2, A):
    from scipy.integrate import simpson

    dlamda = np.deg2rad(lon2) - np.deg2rad(lon1)
    subset = A.sel(longitude=slice(lon1, lon2))
    subset_lambda = np.deg2rad(subset.longitude)
    A_limited = simpson(subset, subset_lambda) / dlamda
    A_limited = xr.DataArray(
        A_limited,
        coords=A.sel(longitude=lon1).coords,
        dims=A.sel(longitude=lon1).dims,
        attrs=A.sel(longitude=lon1).attrs,
        name="limited",
    )
    return A_limited


def plot_single_hadley(
    ds,
    ax,
    lon1,
    lon2,
    skip,
    vskip,
    cmap,
    vmax,
    topo,
    scale=1e3,
    add_key=False,
):
    u_limited = lambda_average(lon1, lon2, ds["u_y"])
    omega_lim = lambda_average(lon1, lon2, ds["omega_y"])
    mx_y_limited = lambda_average(lon1, lon2, ds["m_y"])

    topo = topo.assign_coords(longitude=((topo.longitude + 180) % 360 - 180))
    u_limited = cut_topo(
        u_limited, topo.sel(longitude=slice(lon1, lon2)).min(dim="longitude") / 100
    )
    omega_lim = cut_topo(
        omega_lim, topo.sel(longitude=slice(lon1, lon2)).min(dim="longitude") / 100
    )
    mx_y_limited = cut_topo(
        mx_y_limited, topo.sel(longitude=slice(lon1, lon2)).min(dim="longitude") / 100
    )

    im = ax.pcolormesh(
        mx_y_limited.latitude,
        mx_y_limited.level,
        mx_y_limited,
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap,
    )

    ax.contour(
        mx_y_limited.latitude,
        mx_y_limited.level,
        mx_y_limited,
        levels=[0],
        colors="black",
        linewidths=1,
    )

    # Grey out orography
    ax.contourf(
        mx_y_limited.latitude,
        mx_y_limited.level,
        np.where(np.isnan(mx_y_limited), 0, np.nan),
        levels=[0, 1],
        colors="grey",
    )

    q = ax.quiver(
        mx_y_limited.latitude[::skip],
        mx_y_limited.level[::vskip],
        u_limited[::vskip, ::skip] * 0.05,
        -1 * omega_lim[::vskip, ::skip] * 5e2,
        scale=scale,
    )

    if add_key:
        ax.quiverkey(q, 1.15, 0.75, 25, f"{5} " + "m s$^{-1}$")
        ax.quiverkey(q, 1.15, 0.25, -25, f"{5} " + "hPa s$^{-1}$", angle=90)

    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.minorticks_off()
    xticks = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    yticks = [100, 200, 300, 500, 700, 1000]
    ax.set_yticks([100, 200, 300, 500, 700, 1000])
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [
            "80S",
            "70S",
            "60S",
            "50S",
            "40S",
            "30S",
            "20S",
            "10S",
            "0",
            "10N",
            "20N",
            "30N",
            "40N",
            "50N",
            "60N",
            "70N",
            "80N",
        ]
    )
    ax.set_yticklabels([str(p) for p in yticks])

    return im


if __name__ == "__main__":
    main()
