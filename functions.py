from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats

# import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned

params = {
    "text.usetex": True,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
}
matplotlib.rcParams.update(params)


def query_ned(objname, radius=0.6):
    result_table = Ned.query_object(objname)
    ra = result_table["RA"][0]
    dec = result_table["DEC"][0]
    ra_start = round(ra - radius, 2)
    ra_end = round(ra + radius, 2)
    dec_start = round(dec - radius, 2)
    dec_end = round(dec + radius, 2)
    return ra_start, ra_end, dec_start, dec_end


def query_gaia(
    objname="",
    parallax_start=0,
    parralax_end=10000,
    pmra_start=-1000,
    pmra_end=1000,
    pmdec_start=-1000,
    pmdec_end=1000,
    ra=0.0,
    dec=0.0,
    ruwe=100,
    radius=0.6,
):
    try:
        ra_start, ra_end, dec_start, dec_end = query_ned(objname, radius)
    except:
        ra_start = ra - radius
        ra_end = ra + radius
        dec_start = dec - radius
        dec_end = dec + radius
    # Query Gaia database
    query = """SELECT top 10000 \
    source_id, ra, dec, parallax, phot_g_mean_mag, \
    phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec, radial_velocity \
    FROM gaiadr3.gaia_source \
    WHERE ra between {} and {} \
    AND dec between {} and {} \
    AND abs(pmra_error/pmra)<0.10 \
    AND abs(pmdec_error/pmdec)<0.10 \
    and parallax_over_error > 10 \
    and parallax between {} and {} \
    and pmra between {} and {} \
    and pmdec between {} and {} \
    and ruwe < {} \
    order by parallax desc""".format(
        ra_start,
        ra_end,
        dec_start,
        dec_end,
        parallax_start,
        parralax_end,
        pmra_start,
        pmra_end,
        pmdec_start,
        pmdec_end,
        ruwe,
    )
    job = Gaia.launch_job_async(query)
    r = job.get_results()
    return r


def plot_appmag_vs_dist(objname, r):
    dist = 1000 / r["parallax"]
    appmag = r["phot_g_mean_mag"]
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(np.log10(dist), appmag, s=0.3, c="slateblue")
    plt.xlabel("log$_{10}$[Distance] (parsecs)")
    plt.title(f"Apparent Magnitude vs Distance for {objname}", pad=10, fontsize=15)
    plt.ylabel("Apparent Magnitude in G band")
    plt.savefig(f"{objname}_appmag.png")
    return dist


def gauss(x, A, m, s, c):
    return A * np.exp(-((x - m) ** 2) / (2 * s**2)) + c


def gauss_fit(objname, dist):
    n, bins = np.histogram(dist, bins=100)
    bins = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
    p0 = [max(n), np.mean(dist), np.std(dist), 50]
    popt, pcov = curve_fit(gauss, bins, n, p0=p0)
    x = np.linspace(min(bins), max(bins), 200)
    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(x, gauss(x, *popt), color="salmon", label="Gaussian Fit")
    plt.hist(dist, bins=100, label="Data", color="slateblue")
    plt.title(f"Distance Distribution of {objname}", pad=10, fontsize=15)
    plt.legend()
    plt.xlabel("Distance (parsecs)")
    plt.ylabel("Number of stars")
    plt.savefig(f"{objname}_histfit.png")
    return popt


def parallax_cut(dist, popt):
    dist = dist[
        np.logical_and(
            dist > (popt[1] - 3 * np.abs(popt[2])),
            dist < (popt[1] + 3 * np.abs(popt[2])),
        )
    ]
    parallax_start = 1000 / np.max(dist)
    parallax_end = 1000 / np.min(dist)
    return parallax_start, parallax_end


def plot_pm(objname, parallax_start, parallax_end, clip=0.5, ra=0.0, dec=0.0):
    r = query_gaia(
        objname,
        parallax_start=parallax_start,
        parralax_end=parallax_end,
        ra=ra,
        dec=dec,
    )
    mean_ra = np.mean(r["pmra"])
    mean_dec = np.mean(r["pmdec"])
    xlim = (mean_ra - 10, mean_ra + 10)
    ylim = (mean_dec - 7, mean_dec + 7)
    vel = np.sqrt((r["pmra"] - mean_ra) ** 2 + (r["pmdec"] - mean_dec) ** 2)
    std_vel = np.std(vel, ddof=1)
    r = r[vel < clip * std_vel]
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(r["pmra"], r["pmdec"], s=0.2, color="slateblue")
    plt.xlabel("Proper Motion in RA (mas/yr)")
    plt.ylabel("Proper Motion in Dec (mas/yr)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(f"Proper Motion of {objname}", pad=10, fontsize=15)
    plt.savefig(f"{objname}_pm.png")
    return r


def hr_diag(objname, r, plot=True):
    dist = 1000 / r["parallax"]
    abs_mag = r["phot_g_mean_mag"] - 5 * np.log10(dist) + 5
    bprp = r["phot_bp_mean_mag"] - r["phot_rp_mean_mag"]
    abs_mag = abs_mag[~np.isnan(bprp)]
    bprp = bprp[~np.isnan(bprp)]
    stack = np.vstack((np.copy(bprp), np.copy(abs_mag)))
    kde = gaussian_kde(stack)(stack)
    if plot:
        plt.figure(figsize=(6, 7), dpi=300)
        plt.title(f"HR Diagram of {objname}", pad=10, fontsize=15)
        plt.xlabel("BP-RP")
        plt.ylabel("Absolute Magnitude in G band")
        plt.scatter(bprp, abs_mag, c=kde, s=0.2, cmap="gist_heat")
        plt.colorbar(label="Density")
        plt.ylim(-2, 15)
        plt.vlines(-0.6, 15, -2, color="slateblue", label="O", linewidth=0.5)
        plt.vlines(-0.4, 15, -2, color="slateblue", label="B", linewidth=0.5)
        plt.vlines(0.0, 15, -2, color="slateblue", label="A", linewidth=0.5)
        plt.vlines(0.38, 15, -2, color="slateblue", label="F", linewidth=0.5)
        plt.vlines(0.74, 15, -2, color="slateblue", label="G", linewidth=0.5)
        plt.vlines(1.13, 15, -2, color="slateblue", label="K", linewidth=0.5)
        plt.vlines(2.31, 15, -2, color="slateblue", label="M", linewidth=0.5)
        plt.vlines(4.3, 15, -2, color="slateblue", label="END", linewidth=0.5)
        plt.text(-0.55, -1.6, "O", fontsize=8)
        plt.text(-0.25, -1.6, "B", fontsize=8)
        plt.text(0.14, -1.6, "A", fontsize=8)
        plt.text(0.51, -1.6, "F", fontsize=8)
        plt.text(0.885, -1.6, "G", fontsize=8)
        plt.text(1.67, -1.6, "K", fontsize=8)
        plt.text(3.355, -1.6, "M", fontsize=8)
        plt.gca().invert_yaxis()
        plt.savefig(f"{objname}_hr.png")
    return bprp, abs_mag


def perc_in_spec_class(objname, bprp):
    classes = [-0.6, -0.4, 0.0, 0.38, 0.74, 1.13, 2.31, 4.3]
    number, bins = np.histogram(bprp, bins=classes)
    print(
        f"Percentage of O type stars in {objname} is {round(number[0]/len(bprp)*100, 2)}%"
    )
    print(
        f"Percentage of B type stars in {objname} is {round(number[1]/len(bprp)*100, 2)}%"
    )
    print(
        f"Percentage of A type stars in {objname} is {round(number[2]/len(bprp)*100, 2)}%"
    )
    print(
        f"Percentage of F type stars in {objname} is {round(number[3]/len(bprp)*100, 2)}%"
    )
    print(
        f"Percentage of G type stars in {objname} is {round(number[4]/len(bprp)*100, 2)}%"
    )
    print(
        f"Percentage of K type stars in {objname} is {round(number[5]/len(bprp)*100, 2)}%"
    )
    print(
        f"Percentage of M type stars in {objname} is {round(number[6]/len(bprp)*100, 2)}%"
    )


def plot_all(objname):
    r = query_gaia(objname)
    dist = plot_appmag_vs_dist(objname, r)
    popt = gauss_fit(objname, dist)
    parallax_start, parallax_end = parallax_cut(dist, popt)
    r = plot_pm(objname, parallax_start, parallax_end)
    bprp, _ = hr_diag(objname, r)
    perc_in_spec_class(objname, bprp)
    plt.show()


def sample_from_isochrone(
    isochrone_file="/home/ravioli/astro/ksp/gaia/isochrone.dat",
    n_samples=50,
    plot=True,
    filter=True,
):
    isochrone = pd.read_csv(isochrone_file, sep="\s+")
    isochrone = pd.DataFrame(isochrone)
    # filtering red giants
    if filter:
        isochrone = isochrone[isochrone["logg"] > 3]
    bprp = np.array(isochrone["G_BPmag"] - isochrone["G_RPmag"])
    g = np.array(isochrone["Gmag"])
    random_index = np.random.randint(0, len(isochrone), n_samples)
    isochrone_new = isochrone.iloc[random_index]
    bprp_new = np.array(isochrone_new["G_BPmag"] - isochrone_new["G_RPmag"])
    g_new = np.array(isochrone_new["Gmag"])

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(bprp, g, s=1.5, c="slateblue", label="All stars")
        plt.scatter(
            bprp_new,
            g_new,
            s=1.5,
            c="salmon",
            label=f"Randomly selected {n_samples} stars",
        )
        plt.gca().invert_yaxis()
        plt.xlabel("BP-RP")
        plt.ylabel("G")
        plt.title("Gaia DR3 isochrone")
        plt.legend()
        plt.show()

    return isochrone_new, bprp, g, isochrone


def powerlaw(x, a, gamma):
    return a * (x) ** (-gamma)
