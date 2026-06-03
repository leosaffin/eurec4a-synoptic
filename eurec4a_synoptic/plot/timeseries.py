import datetime

from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import label_axes

limits = [
    datetime.datetime(2020, 1, 20),
    datetime.datetime(2020, 1, 25),
    datetime.datetime(2020, 1, 29),
    datetime.datetime(2020, 2, 8),
    datetime.datetime(2020, 2, 15),
]

# color_dict = {
#     "Fish": "C0",
#     "Flowers": "C1",
#     "Gravel": "C7",
#     "Sugar": "w",
#     "Unclassified": "w",
# }

color_dict = {
    "Fish": np.array([232, 198, 33]) / 255,
    "Flowers": np.array([249, 103, 103]) / 255,
    "Gravel": np.array([53, 252, 213]) / 255,
    "Sugar": np.array([92, 226, 253]) / 255,
    "Unclassified": "w",
}


def main():
    context = xr.open_dataset(
        "processed_data/EUREC4A_ManualClassifications_l3_VIS_daily.zarr"
    )
    data_bco = context.freq.sel(
        latitude=13.1626389, longitude=-059.4287500, method="nearest"
    )
    data_bco = data_bco.sel(date=slice("2020-01-20", "2020-02-14"))

    grimm_data_2d = xr.open_dataarray("grimm.nc")
    concentration = grimm_data_2d.sum(dim="particle_size_bin_lower_boundary") / 1000
    concentration = concentration.resample(time="h").mean()

    bco = xr.open_dataset("bco.nc")
    cf = xr.open_dataarray("cloud_fraction.nc")

    frequency = np.zeros((len(data_bco.date)))

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex="col", layout="constrained")

    for p in ["Sugar", "Gravel", "Fish", "Flowers", "Unclassified"]:
        if p == "Unclassified":
            hatch = "xxxxx"
        else:
            hatch = None
        data_ = data_bco.sel(pattern=p).values * 100
        data_[np.isnan(data_)] = 0
        axes[0].bar(
            dates.date2num(
                pd.to_datetime(data_bco.date) + datetime.timedelta(hours=12)
            ),
            data_,
            label=p,
            bottom=frequency,
            color=color_dict[p],
            ec="k",
            hatch=hatch,
        )

        frequency += data_

    axes[0].legend(bbox_to_anchor=(1, 0.85))
    axes[0].set_ylim(0, 125)
    axes[0].set_ylabel("agreement / %")

    im = axes[1].pcolormesh(
        cf.time, cf.range, np.ma.masked_where(cf == 0, cf).transpose(), vmin=0, vmax=1
    )
    plt.colorbar(im, pad=-0.2, label="Cloud Fraction")
    axes[1].set_ylabel("Height (m)")

    axes[2].plot(
        concentration.time,
        concentration,
        "-k",
    )
    axes[2].set_ylim(0, 200)
    axes[2].set_ylabel("Particle Concentration\n0.25-32$\mathrm{\mu}$m (#/cc)")

    dt = bco.time[1] - bco.time[0]
    axes[3].bar(
        x=bco.time + 0.5 * dt, height=bco.R, width=dt, edgecolor="k", color="C0"
    )
    axes[3].set_ylim(0, 5)
    axes[3].set_ylabel("Rain Amount (mm)", color="C0")

    # Following https://matplotlib.org/stable/gallery/spines/multiple_yaxis_with_spines.html
    twin1 = axes[3].twinx()
    twin2 = axes[3].twinx()
    twin2.spines.right.set_position(("axes", 1.1))
    twin2.tick_params(axis="y", colors="C4")

    twin1.plot(bco.time, bco.VEL, "-k")
    twin1.set_ylabel("Relative Humidity (%)")
    twin1.set_ylim(0, 10)
    twin1.set_ylabel("Wind Speed (m s$^{-1}$)")

    twin2.plot(bco.time, bco.DIR, "--C4")
    twin2.set_ylim(120, 60)
    twin2.set_ylabel(r"Wind Direction ($^{\circ}$)", color="C4")

    # Highlight the different time periods on each axis
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        for n in range(4):
            ax.fill_between(
                [limits[n], limits[n + 1]], ymin, ymax, alpha=0.25, zorder=-1
            )
        ax.set_ylim([ymin, ymax])
    axes[1].xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
    axes[1].set_xticks(limits)
    axes[1].set_xticks(pd.date_range(limits[0], limits[-1], freq="D"), minor=True)
    axes[1].set_xlim(limits[0], limits[-1])

    fig.autofmt_xdate()

    label_axes(axes)

    plt.savefig("fig1_timeseries.pdf")
    plt.show()


def bco_weather(wxt, variables, ax, sampling="6h"):
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

        ax.plot(y.time, y, color=f"C{n}")
        ax.set_ylabel(wxt[var].attrs["standard_name"], color=f"C{n}")
        if n == 0:
            ax = ax.twinx()


if __name__ == "__main__":
    main()
