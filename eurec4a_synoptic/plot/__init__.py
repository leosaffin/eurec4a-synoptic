from string import ascii_lowercase

from cartopy.crs import EqualEarth, PlateCarree
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np


projection = EqualEarth()
transform = PlateCarree()

extent = [-75, 0, 0, 40]


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


def mslp_overlay(ax, mslp, levels=np.arange(958, 1024 + 6, 6)):
    lines = ax.contour(
        mslp.longitude,
        mslp.latitude,
        mslp,
        levels,
        transform=transform,
        colors="k",
        linewidths=2,
    )
    clabels = ax.clabel(lines, inline=False, fontsize=5, fmt=lambda x: f"{x:.0f} hPa")
    for txt in clabels:
        txt.set_path_effects(
            [
                patheffects.Stroke(linewidth=3, foreground="white"),
                patheffects.Normal(),
            ]
        )
