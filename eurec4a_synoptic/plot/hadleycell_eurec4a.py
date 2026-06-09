import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.crs import EqualEarth, PlateCarree
import matplotlib.patheffects as path_effects


def lon_convert(ds, cut=False):
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180))
    # Sort longitudes to maintain order from -180 to 180
    ds = ds.sortby("longitude")
    if cut == True:
        ds = ds.sel(longitude=slice(-30, 120), latitude=slice(45, -45))
    return ds


def cut_topo(ds, surface_pressure):
    """
    Masks out values where the pressure level is above the surface pressure (i.e., underneath the topography).

    Parameters:
    ds (xr.Dataset): Dataset containing variables on pressure levels (dimensions: level, latitude, longitude).
    surface_pressure (xr.DataArray): Mean surface pressure (hPa) with dimensions (latitude, longitude).
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
    ds, ax, lon1, lon2, skip, vskip, cmap, vmax, num_colors, topo, scale=2.3
):
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180))
    ds = ds.sortby(ds.longitude).sel(latitude=slice(40, -45))
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

    vmin = -vmax
    levels = np.linspace(vmin, vmax, num_colors)
    contour = ax.contourf(
        mx_y_limited.latitude,
        mx_y_limited.level,
        mx_y_limited,
        levels=levels,
        cmap=cmap,
    )
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ax.set_xticks(
        [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    )
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
    ax.set_yticklabels(
        [f"{p}" for p in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]]
    )
    ax.contour(
        mx_y_limited.latitude,
        mx_y_limited.level,
        mx_y_limited,
        levels=[0],
        colors="black",
        linewidths=1,
    )

    Q = ax.quiver(
        mx_y_limited.latitude[::skip],
        mx_y_limited.level[::vskip],
        1 * u_limited[::vskip, ::skip] / 100 / 100,
        -1 * omega_lim[::vskip, ::skip],
        scale=scale,
        width=0.2e-2,
    )
    ax.set_ylabel("p (hPa)", fontsize=16)
    ax.tick_params(
        direction="out",
        length=8,
        width=2,
        colors="black",
        labelsize=12,
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    return contour


#'---mean local haldey circulation map------------------------------------------------------------------------------------------------------------------'

mean_eureca = xr.open_dataset("mean_m_y_eureca-timepsan.nc")

projection = EqualEarth()
transform = PlateCarree()

vmax = 0.015
vmin = -vmax
levels = np.linspace(vmin, vmax, 31)

mean_eureca = lon_convert(mean_eureca)


# # Definition of function to add circle after plotting figure
def add_halo_circle(
    ax, lon=-(57 + (43 / 60)), lat=13 + (18 / 60), radius=1, *args, **kwargs
):
    angles = np.arange(0, 360 + 1)
    lons = lon + np.cos(np.deg2rad(angles))
    lats = lat + np.sin(np.deg2rad(angles))

    ax.plot(lons, lats, color="teal", *args, **kwargs)


# # Run before plotting the figures
fig, ax = plt.subplots(
    1,
    figsize=(16, 12),
    sharex="all",
    sharey="all",
    subplot_kw=dict(projection=projection),
)

mslp = xr.open_dataset("mslp.nc")
mean_mslp = mslp.mean(dim="time").compute()
mean_mslp = (
    mean_mslp.assign_coords(longitude=((mean_mslp.longitude + 180) % 360 - 180))
    .sortby("longitude")
    .sel(longitude=slice(-110, 0), latitude=slice(45, 0))
)
mean_mslp = mean_mslp.mean_sea_level_pressure

lines = ax.contour(
    mean_mslp.longitude,
    mean_mslp.latitude,
    mean_mslp,
    transform=ccrs.PlateCarree(),
    colors="k",
    linewidths=2,
)
contour = ax.contourf(
    mean_eureca.longitude,
    mean_eureca.latitude,
    mean_eureca.m_y.sel(level=500),
    transform=ccrs.PlateCarree(),
    levels=levels,
    cmap="seismic",
)
clabels = ax.clabel(lines, inline=False, fontsize=8, fmt=lambda x: f"{x / 100:.0f} hPa")
for txt in clabels:
    txt.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    )
ax.coastlines(resolution="10m")
ax.gridlines(draw_labels=["left", "bottom"])
add_halo_circle(ax, transform=transform)
ax.set_extent([-100, 0, 0, 45], crs=transform)

cbar_ax = fig.add_axes([0.10, 0.11, 0.8, 0.04])
fig.colorbar(contour, cax=cbar_ax, orientation="horizontal", label="")
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, hspace=0.1, wspace=0.05)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.88)

plt.savefig(
    "eureca_localhadleycirc_mean-entire-eureca-period", dpi=300, bbox_inches="tight"
)

# ---------------------Anomaly from long term mean--------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(
    1,
    figsize=(16, 12),
    sharex="all",
    sharey="all",
    subplot_kw=dict(projection=projection),
)

ds_l30y = xr.open_dataset("mean_500hPa_m_y_1990-2023.nc")

data = lon_convert(ds_l30y)

contour = ax.contourf(
    data.longitude,
    data.latitude,
    mean_eureca.m_y.sel(level=500) - data.m_y,
    transform=ccrs.PlateCarree(),
    levels=levels,
    cmap="seismic",
)
lines = ax.contour(
    mean_mslp.longitude,
    mean_mslp.latitude,
    mean_mslp,
    transform=ccrs.PlateCarree(),
    colors="k",
    linewidths=2,
)
clabels = ax.clabel(lines, inline=False, fontsize=8, fmt=lambda x: f"{x / 100:.0f} hPa")
for txt in clabels:
    txt.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    )

ax.coastlines(resolution="10m")
ax.gridlines(draw_labels=["left", "bottom"])
add_halo_circle(ax, transform=transform)
ax.set_extent([-100, 0, 0, 45], crs=transform)

cbar_ax = fig.add_axes([0.10, 0.11, 0.8, 0.04])
fig.colorbar(contour, cax=cbar_ax, orientation="horizontal", label="")
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, hspace=0.1, wspace=0.05)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.88)


plt.savefig(
    "eureca_localhadleycirc_anomaly-entire-eureca-period", dpi=300, bbox_inches="tight"
)

# ----Mean cross section-------------------------------------------------------------------------------------------------------------------
topo = xr.load_dataarray(f"mean_sfp_1979-2023.nc")

fig, ax = plt.subplots(1, figsize=(9, 7))
contour = plot_single_hadley(
    mean_eureca, ax, -60, -40, 5, 1, "seismic", 0.01, 41, topo=topo, scale=1.4
)
plt.colorbar(contour, orientation="horizontal", aspect=30, pad=0.07)
plt.tight_layout()
fig.savefig(
    "eureca_regionalhadleycirc_mean-entire-eureca-period", dpi=300, bbox_inches="tight"
)


# ----four Cross section-------------------------------------------------------------------------------------------------------------------
m_y = xr.open_dataarray(f"m_y_eureca.nc")
u_y = xr.open_dataarray(f"u_y_eureca.nc")
omega_y = xr.open_dataarray(f"omega_y_eureca.nc")

ds = xr.Dataset({"m_y": m_y, "u_y": u_y, "omega_y": omega_y})

ds = ds.rename({"pressure_level": "level"})

fig, axes = plt.subplots(2, 2, figsize=(9 * 2, 7 * 2))
for i, ax in enumerate(axes.flatten()):
    contour = plot_single_hadley(
        ds.isel(valid_time=i), ax, -60, -40, 5, 1, "seismic", 0.05, 41, topo=topo
    )
    ax.set_title(
        str(ds.isel(valid_time=i).valid_time.values)[:10] + " 18:00 UTC", pad=20
    )
cbar_ax = fig.add_axes([0.10, -0.03, 0.8, 0.03])
fig.colorbar(contour, cax=cbar_ax, orientation="horizontal", label="")
plt.tight_layout()

plt.savefig("eureca_regionalhadleycirc_individualtimes", dpi=300, bbox_inches="tight")


# ------------------Four maps-------------------------------------------------------------------------------------------------------------
ds = xr.open_dataarray(f"m_y_eureca.nc").sel(pressure_level=500)
mlsp_1800h = xr.open_dataset("mslp_1800h.nc")

projection = EqualEarth()
transform = PlateCarree()
plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots(
    2,
    2,
    figsize=(16, 12),
    sharex="all",
    sharey="all",
    subplot_kw=dict(projection=projection),
)
ax = ax.flatten()


# Definition of function to add circle after plotting figure
def add_halo_circle(
    ax, lon=-(57 + (43 / 60)), lat=13 + (18 / 60), radius=1, *args, **kwargs
):
    angles = np.arange(0, 360 + 1)
    lons = lon + np.cos(np.deg2rad(angles))
    lats = lat + np.sin(np.deg2rad(angles))

    ax.plot(lons, lats, color="teal", *args, **kwargs)


for i, j in enumerate(ds.valid_time):
    mean_eureca_1d = ds.sel(valid_time=j)
    vmax = 0.20
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 41)

    mean_eureca_1d = mean_eureca_1d.assign_coords(
        longitude=((mean_eureca_1d.longitude + 180) % 360 - 180)
    ).sortby("longitude")
    contour = ax[i].contourf(
        mean_eureca_1d.longitude,
        mean_eureca_1d.latitude,
        mean_eureca_1d,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap="seismic",
    )

    mlsp_1800h_sel = (
        mlsp_1800h.mean_sea_level_pressure.sel(valid_time=j)
        .assign_coords(longitude=((mslp.longitude + 180) % 360 - 180))
        .sortby("longitude")
        .sel(longitude=slice(-110, 0), latitude=slice(45, 0))
    )

    lines = ax[i].contour(
        mlsp_1800h_sel.longitude,
        mlsp_1800h_sel.latitude,
        mlsp_1800h_sel,
        transform=ccrs.PlateCarree(),
        colors="k",
        levels=np.arange(98800, 102400 + 600, 600),
    )
    clabels = ax[i].clabel(
        lines, inline=False, fontsize=8, fmt=lambda x: f"{x / 100:.0f} hPa"
    )
    for txt in clabels:
        txt.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ]
        )
    ax[i].coastlines(resolution="10m")
    ax[i].gridlines(draw_labels=["left", "bottom"])
    add_halo_circle(ax[i], transform=transform)
    ax[i].set_extent([-100, 0, 0, 45], crs=transform)


plt.tight_layout()
cbar_ax = fig.add_axes([0.10, 0.09, 0.8, 0.03])
cbar = fig.colorbar(
    contour, cax=cbar_ax, orientation="horizontal", label="", fraction=0.3
)
cbar.ax.tick_params(labelsize=13)
# fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, hspace=0.1, wspace=0.05)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.88)
#
fig.savefig("map_localhadleycirc_individualtimes.png", dpi=300, bbox_inches="tight")
