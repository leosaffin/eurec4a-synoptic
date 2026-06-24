import datetime
from string import ascii_lowercase

import cmcrameri
import iris
import iris.plot as iplt
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from metpy.plots import SkewT
import numpy as np
import pandas as pd
import xarray as xr

from . import add_halo_circle, label_axes, transform, projection, extent, mslp_overlay

bounds = np.arange(-2, 2.1, 0.25)
bounds = bounds[bounds != 0]
norm_pv_low = BoundaryNorm(
    boundaries=[-2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2], ncolors=256
)
norm_pv_high = BoundaryNorm(boundaries=np.arange(0, 10 + 1), ncolors=256)
norm_humidity = BoundaryNorm(boundaries=np.arange(0, 0.021, 0.0025), ncolors=256)
# Nonlinear colour scale for AOD

norm_aod = BoundaryNorm(boundaries=[0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2], ncolors=256)


def main():
    era5 = iris.load("era5_eurec4a.nc")
    era5.extract_cube("Potential vorticity").convert_units("PVU")
    aod = iris.load_cube("merra2_aod_eurec4a.nc")

    for n in range(500):
        time = datetime.datetime(2020, 1, 20, 18) + datetime.timedelta(days=n)
        four_panel_figure(era5, aod, time)
        plt.savefig(f"eurec4a-snapshot_{time.strftime('%Y%m%d%H%M')}.pdf")
        plt.close()


def four_panel_figure(cubes, dust, time, winds_ratio=3):

    fig, axes = plt.subplot_mosaic(
        """
        12
        ab
        --
        34
        cd
        ,,
        56
        """,
        figsize=(8, 11),
        height_ratios=[10, 1, 1, 10, 1, 1, 10],
        per_subplot_kw={str(n): dict(projection=projection) for n in range(1, 5 + 1)},
    )
    for ax in ["-", ","]:
        axes[ax].set_axis_off()

    axes["6"].set_axis_off()
    # ax_skewt = SkewT(fig=fig, subplot=(4, 2, 8))
    ax_skewt = SkewT(fig=fig, subplot=GridSpec(34, 10)[25:, 6:10])
    ax_skewt.ax.yaxis.set_label_position("right")
    ax_skewt.ax.yaxis.tick_right()
    plot_skewt(ax_skewt, time)

    fig.suptitle(time.strftime("%Y-%m-%d %H:%M"))

    time_cs = iris.Constraint(time=lambda cell: cell.point == time)
    cubes = cubes.extract(time_cs)
    for n, (variable, plev, kwargs, scale, example) in enumerate(
        [
            (
                "Potential vorticity",
                950,
                dict(vmin=-2, vmax=2, cmap="cmc.vik"),
                3e-5,
                15,
            ),
            (
                "Potential vorticity",
                250,
                dict(vmin=0, vmax=10, cmap="cmc.batlowW_r"),
                1e-4,
                50,
            ),
            (
                "specific_humidity",
                850,
                dict(vmin=0, vmax=0.02, cmap="cmc.oslo_r"),
                5e-5,
                25,
            ),
        ],
        start=1,
    ):
        cube = cubes.extract_cube(iris.Constraint(variable, pressure_level=plev))
        lon = cube.coord(axis="x", dim_coords=True).points
        lat = cube.coord(axis="y", dim_coords=True).points
        u = cubes.extract_cube(
            iris.Constraint("eastward_wind", pressure_level=plev)
        ).data
        v = cubes.extract_cube(
            iris.Constraint("northward_wind", pressure_level=plev)
        ).data

        pcolormesh_with_winds(
            axes[str(n)],
            axes[ascii_lowercase[n - 1]],
            cube,
            lon,
            lat,
            u,
            v,
            scale,
            example,
            **kwargs,
        )

    ax = plt.axes(axes["4"])
    aod = dust.extract(time_cs)
    im = iplt.pcolormesh(aod, norm=norm_aod, cmap="cmc.bilbao_r")
    plt.colorbar(im, cax=axes["d"], orientation="horizontal")

    for n in range(1, 4 + 1):
        axes[str(n)].set_extent(extent, crs=transform)

    mslp = cubes.extract_cube("air_pressure_at_mean_sea_level")
    mslp.convert_units("hPa")
    mslp = xr.DataArray.from_iris(mslp)
    mslp_overlay(ax, mslp)

    ax.coastlines(resolution="10m")
    add_halo_circle(ax, transform=transform)

    plot_goes(axes["5"], time)

    axes["1"].set_title("PV at 950hPa (PVU)")
    axes["2"].set_title("PV at 250hPa (PVU)")
    axes["3"].set_title("Specific Humidity\nat 850hPa (kg kg$^{-1}$)")
    axes["4"].set_title("MSLP (hPa) and AOD")
    axes["5"].set_title("GOES Geocolor")

    axes["1"].gridlines(draw_labels=["left"])
    axes["2"].gridlines()
    axes["3"].gridlines(draw_labels=["left", "bottom"])
    axes["4"].gridlines(draw_labels=["bottom"])
    axes["5"].gridlines(draw_labels=["left", "bottom"])

    label_axes([axes[str(ax)] for ax in range(1, 6 + 1)])


def pcolormesh_with_winds(ax, cax, cube, lon, lat, u, v, scale, example, **kwargs):
    ax = plt.axes(ax)
    im = iplt.pcolormesh(cube, **kwargs)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    ax.coastlines(resolution="10m")
    add_halo_circle(ax, transform=transform)

    q = plt.quiver(
        lon,
        lat,
        u,
        v,
        transform=transform,
        regrid_shape=20,
        angles="xy",
        scale_units="xy",
        scale=scale,
    )
    plt.quiverkey(q, 0.95, 1.05, example, f"{example} " + "m s$^{-1}$")


def plot_goes(ax, time):
    rgb = xr.open_dataset(f"goes_rgb_era5_grid_{time.strftime('%Y%m%dT%H%M')}.nc")

    ax.imshow(
        get_TrueColor_RGB(rgb),
        origin="lower",
        extent=[
            rgb.longitude.values.min(),
            rgb.longitude.values.max(),
            rgb.latitude.values.min(),
            rgb.latitude.values.max(),
        ],
        transform=transform,
        regrid_shape=len(rgb.longitude) * 2,
    )
    ax.set_extent([-60, -30, 7.5, 25], crs=transform)
    ax.coastlines(resolution="10m")

    add_halo_circle(ax, transform=transform, fill=False)


def get_TrueColor_RGB(C):
    """
    Taken from
    https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html
    """
    R = C["CMI_C02"].data
    G = C["CMI_C03"].data
    B = C["CMI_C01"].data

    # Apply range limits for each channel. RGB values must be between 0 and 1
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    # Apply a gamma correction to the image
    gamma = 2.2
    R = np.power(R, 1 / gamma)
    G = np.power(G, 1 / gamma)
    B = np.power(B, 1 / gamma)

    # Calculate the "True" Green
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.maximum(G_true, 0)
    G_true = np.minimum(G_true, 1)

    return np.dstack([R, G_true, B])


def plot_skewt(skewt, time):
    df = pd.read_csv(f"grantley_adams_sounding_{time.strftime('%Y%m%d')}T1200.csv")

    skewt.plot(df["pressure_hPa"], df["temperature_C"], "-C1", label="$T$")
    skewt.plot(df["pressure_hPa"], df["dew point temperature_C"], "-C0", label="$T_d$")
    skewt.plot_dry_adiabats(colors="k", linestyle="-", lw=0.5, alpha=1)
    skewt.plot_moist_adiabats(colors="k", linestyle="--", lw=0.5, alpha=1)

    u = -df["wind speed_m/s"] * np.sin(np.deg2rad(df["wind direction_degree"]))
    v = -df["wind speed_m/s"] * np.cos(np.deg2rad(df["wind direction_degree"]))
    wind_levs = np.exp(np.linspace(np.log(1000), np.log(300), 15))
    # wind_levs = np.array([1000, 900, 800, 700, 600, 500, 450, 400, 350, 300])

    ui = np.interp(wind_levs[::-1], df["pressure_hPa"][::-1], u[::-1])[::-1]
    vi = np.interp(wind_levs[::-1], df["pressure_hPa"][::-1], v[::-1])[::-1]

    skewt.plot_barbs(wind_levs, ui, vi, xloc=-0.15, x_clip_radius=1)

    skewt.ax.legend(loc="upper right")
    skewt.ax.set_xlim(-40, 30)
    skewt.ax.set_ylim(1050, 250)
    skewt.ax.set_xlabel(r"Temperature ($^{\circ}$C)")
    skewt.ax.set_ylabel("Pressure (hPa)")
    skewt.ax.set_title("Grantley Adams Sounding")


if __name__ == "__main__":
    main()
