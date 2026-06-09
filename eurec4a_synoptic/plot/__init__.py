from string import ascii_lowercase

from cartopy.crs import EqualEarth, PlateCarree
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np


projection = EqualEarth()
transform = PlateCarree()

extent = [-75, 0, 0, 40]

snapshot_times = np.array(
    ["2020-01-22", "2020-01-27", "2020-02-05", "2020-02-12"], dtype="datetime64"
)


def label_axes(axes, xpos=0.01, ypos=1.025):
    for n, ax in enumerate(axes):
        ax.text(xpos, ypos, f"({ascii_lowercase[n]})", transform=ax.transAxes)


def add_halo_circle(
    ax, lon=-(57 + (43 / 60)), lat=13 + (18 / 60), radius=1, *args, **kwargs
):
    circle = plt.Circle((lon, lat), radius=radius, color="C6", *args, **kwargs)
    ax.add_patch(circle)


def roll_lons(ds):
    ds = ds.roll(longitude=int(len(ds.longitude) / 2), roll_coords=True)
    return ds.assign(longitude=((ds.longitude + 180) % 360) - 180)


def mslp_overlay(ax, mslp, lmin=952, lmax=1040, small_step=1, big_step=6):
    ax.contour(
        mslp.longitude,
        mslp.latitude,
        mslp,
        range(lmin, lmax + small_step, small_step),
        colors="k",
        linewidths=0.5,
        transform=transform,
    )
    lines = ax.contour(
        mslp.longitude,
        mslp.latitude,
        mslp,
        range(lmin, lmax + big_step, big_step),
        transform=transform,
        colors="k",
        linewidths=1.5,
        linestyles="--",
    )
    clabels = ax.clabel(lines, inline=False, fontsize=5, fmt=lambda x: f"{x:.0f} hPa")
    for txt in clabels:
        txt.set_path_effects(
            [
                patheffects.Stroke(linewidth=3, foreground="white"),
                patheffects.Normal(),
            ]
        )
