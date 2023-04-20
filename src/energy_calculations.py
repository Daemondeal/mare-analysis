from tkinter import W
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import numpy as np
import time
import math

from matplotlib.dates import DateFormatter, MonthLocator
from numpy import interp, ndarray

from utils import read_csv, write_dataframe

AIR_DENSITY = 1.225  # kg/m^3
ENERGY_CONSUMED_YEARLY_WALLIS = 15.4E9  # 15.4 GWh (2015)
ENERGY_CONSUMED_YEARLY_LIPARI = 34.8E9  # 34.8 GWh (2021)


class WindTurbine:
    rotor_diameter: float  # In meters
    cut_in_speed: float  # In meters per second
    cut_out_speed: float  # In meters per second
    rated_speed: float  # In meters per second
    rated_power: float  # In Watts
    # cp_data: list[(float, float)]  # Data containing values for cp

    def __init__(self, rotor_diameter, cut_in_speed, cut_out_speed, rated_speed, rated_power, cp_file):
        self.rotor_diameter = rotor_diameter
        self.cut_in_speed = cut_in_speed
        self.cut_out_speed = cut_out_speed
        self.rated_speed = rated_speed
        self.rated_power = rated_power

        self.cp_data = []
        cp_dataframe = read_csv(cp_file, ignore=["Power", "Thrust", "Ct"])
        for i in range(len(cp_dataframe["Wind Speed"])):
            self.cp_data.append((cp_dataframe["Wind Speed"][i], cp_dataframe["Cp"][i]))

        self.cp_func = interpolate.interp1d(cp_dataframe["Wind Speed"], cp_dataframe["Cp"], kind="cubic")

    def get_closest_cp(self, wind_speed: float) -> float:
        min_dist = 1e9
        value = 0
        for wind, cp in self.cp_data:
            if abs(wind - wind_speed) < min_dist:
                min_dist = abs(wind - wind_speed)
                value = cp

        return value

    def get_cp(self, wind_speed):
        return self.cp_func(wind_speed)

    # Equation found in "Materiale Esercitazioni/Esercitazione 13_05/Esercitazione_produttivita.pdf"
    def power_generated(self, wind_speed: float) -> float:
        if wind_speed < self.cut_in_speed:
            return 0
        elif self.rated_speed > wind_speed > self.cut_in_speed:
            rotor_area = math.pi * (self.rotor_diameter / 2) ** 2
            power_coefficient = self.get_cp(wind_speed)

            return 0.5 * AIR_DENSITY * rotor_area * power_coefficient * (wind_speed ** 3)
        elif self.cut_out_speed > wind_speed > self.rated_speed:
            return self.rated_power
        elif wind_speed > self.cut_out_speed:
            return 0

    def power_ratio(self, wind_speed: float) -> float:
        return self.power_generated(wind_speed) / self.rated_power


class WecPowerMatrix:
    def __init__(self, filename: str):
        self.mat = []
        with open(filename, "r") as matrix_file:
            heading = matrix_file.readline().strip().split(",")[1:]

            for item in heading:
                self.mat.append((float(item), []))

            for line in matrix_file:
                data = line.strip().split(",")
                point = float(data[0])

                for i, col in enumerate(self.mat):
                    # Data is in kW, so I need to convert it to Watts
                    col[1].append((point, float(data[i + 1]) * 1_000))

    def get_power(self, height: float, period: float) -> float:
        _, height_row = min(self.mat, key=lambda x: abs(height - x[0]))
        _, point = min(height_row, key=lambda x: abs(period - x[0]))

        return point

    def get_max_power(self) -> float:
        maximum = 0

        for _, row in self.mat:
            for _, power in row:
                if power > maximum:
                    maximum = power

        return maximum


def plot_data_time(x, y):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, y)

    y_line = [12 for x_ in x]
    ax.plot(x, y_line)

    y_line = [9.75 for x_ in x]
    ax.plot(x, y_line)

    # Tick Every Month
    ax.xaxis.set_major_locator(MonthLocator())

    # Show date as month abbreviation
    ax.xaxis.set_major_formatter(DateFormatter("%b"))

    ax.set_ylabel("Wind [m/s]")

    # ax.autoscale(enable=True, axis="both", tight=True)
    # fig.show()
    fig.savefig("../plots/average_wind.png")


def plot_data(x, y):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, y)
    ax.grid()

    ax.set_xlabel("Test")

    # ax.invert_yaxis()

    # ax.autoscale(enable=True, axis="both", tight=True)
    fig.show()
    # fig.savefig("../plots/turbine_generation")


def generate_cf_files():
    # turbine = WindTurbine(
    #     rotor_diameter=126,
    #     cut_in_speed=3,
    #     cut_out_speed=25,
    #     rated_speed=11.4,
    #     rated_power=5_000_000,  # 5 MW
    #     cp_file="../data/turbine_cp.csv"
    # )

    land_turbine = WindTurbine(
        rotor_diameter=150,
        cut_in_speed=3.25,
        cut_out_speed=25,
        rated_speed=9.75,
        rated_power=4_000_000,
        cp_file="../data/land_turbine.csv"
    )

    offshore_turbine = WindTurbine(
        rotor_diameter=155,
        cut_in_speed=4,
        cut_out_speed=25,
        rated_speed=12,
        rated_power=6_000_000,  # 6 MW
        cp_file="../data/6mw_turbine.csv"
    )

    turbine = offshore_turbine

    wec_matrix = WecPowerMatrix("../data/wec_matrix.csv")

    print("Calculating for Wallis...")
    wallis_cf = get_island_cf_and_consumption(
        island_folder="wallis",
        yearly_consumption=ENERGY_CONSUMED_YEARLY_WALLIS,
        turbine=turbine,
        wec_matrix=wec_matrix
    )

    print("Calculating for Lipari...")
    lipari_cf = get_island_cf_and_consumption(
        island_folder="lipari",
        yearly_consumption=ENERGY_CONSUMED_YEARLY_LIPARI,
        turbine=turbine,
        wec_matrix=wec_matrix
    )

    for frame in [wallis_cf, lipari_cf]:
        n = len(frame["cf_fotovoltaico"])
        average_wind_cf = sum(frame["cf_eolico"]) / n
        average_solar_cf = sum(frame["cf_fotovoltaico"]) / n
        average_wec_cf = sum(frame["cf_wec"]) / n

        print(f"Average Wind CF: {average_wind_cf * 100:.2f}%")
        print(f"Average Solar CF: {average_solar_cf * 100:.2f}%")
        print(f"Average WEC CF: {average_wec_cf * 100:.2f}%")
        print()


    print("Writing Results...")
    write_dataframe(f"../results/data_lipari.csv", lipari_cf)
    write_dataframe(f"../results/data_wallis.csv", wallis_cf)
    print("Done!")


def get_island_cf_and_consumption(island_folder: str, yearly_consumption: float,
                                  turbine: WindTurbine, wec_matrix: WecPowerMatrix) -> dict:
    base_folder = f"../data/{island_folder}"

    wind_frame = read_csv(f"{base_folder}/wind_and_waves_fixed.csv")
    solar_frame = read_csv(f"{base_folder}/sun_data_fixed.csv")
    consumption_frame = read_csv(f"../data/normalized_power_consumption.csv")

    time = wind_frame["time"]

    wind_u = np.array(wind_frame["wind_100m_u"])
    wind_v = np.array(wind_frame["wind_100m_v"])

    period = wind_frame["energy_wave_period"]
    height = wind_frame["significant_wave_height"]

    wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
    
    wind_energy = np.array([turbine.power_generated(w) for w in wind_speed])
    solar_energy = np.array(solar_frame["P"])
    wave_energy = np.array([wec_matrix.get_power(p, h) for p, h in zip(period, height)])
    energy_used = np.array(consumption_frame["Normalized Power Consumption"]) * yearly_consumption

    wind_installed = turbine.rated_power
    solar_installed = 1_000  # 1kW
    wave_installed = wec_matrix.get_max_power()

    wind_cf = wind_energy / wind_installed
    wave_cf = wave_energy / wave_installed
    solar_cf = solar_energy / solar_installed

    return {
        "data": [t.isoformat() for t in time],
        "cf_fotovoltaico": solar_cf,
        "cf_eolico": wind_cf,
        "cf_wec": wave_cf,
        "energia_consumata": energy_used
    }


def get_balancing(cf_frame: dict, solar_installed: float, wind_installed: float, wec_installed: float) -> dict:
    cardinality = len(cf_frame["data"])

    wind_cf = np.array(cf_frame["cf_fotovoltaico"])
    solar_cf = np.array(cf_frame["cf_fotovoltaico"])
    wec_cf = np.array(cf_frame["cf_wec"])

    consumption = np.array(cf_frame["energia_consumata"])

    wind = wind_cf * wind_installed
    solar = solar_cf * solar_installed
    wec = wec_cf * wec_installed

    residue = wind + solar + wec - consumption

    return {
        "ora": [x.isoformat() for x in cf_frame["data"]],
        "fotovoltaico_cf": solar_cf,
        "fotovoltaico_produzione": solar,
        "eolico_cf": wind_cf,
        "eolico_produzione": wind,
        "wec_cf": wec_cf,
        "wec_produzione": wec,
        "consumo": consumption,
        "produzione_diesel": [-x if x < 0 else 0 for x in residue]
    }

def generate_energy_balancing():
    wallis_cf = read_csv("../results/data_wallis.csv")
    lipari_cf = read_csv("../results/data_lipari.csv")

    wallis_balancing = get_balancing(
        wallis_cf,
        solar_installed=5_000_000,
        wind_installed=5_000_000,
        wec_installed=5_000_000
    )

    print(wallis_balancing)


def plot_cf(filename: str, cf, *, axis, xlabel: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(axis, weights=[x * 100 for x in cf], bins=np.arange(len(cf) + 1)-0.5, linewidth=1.2, edgecolor="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CF (%)")

    fig.savefig(filename)

def graph_daily_cf(island: str, frame: dict):
    solar_weights = [[] for _ in range(24)]
    wec_weights = [[] for _ in range(24)]
    wind_weights = [[] for _ in range(24)]
    
    for i in range(len(frame["data"]) // 24):
        for j in range(24):
            solar_weights[j].append(frame["cf_fotovoltaico"][i * 24 + j])
            wec_weights[j].append(frame["cf_wec"][i * 24 + j])
            wind_weights[j].append(frame["cf_eolico"][i * 24 + j])

    solar_avg = [sum(x) / len(x) for x in solar_weights]
    wec_avg = [sum(x) / len(x) for x in wec_weights]
    wind_avg = [sum(x) / len(x) for x in wind_weights]

    hours_in_a_day = [f"{x:02}" for x in range(24)]

    plot_cf(f"../plots/{island}/solar_cf_hourly.png", solar_avg, axis=hours_in_a_day, xlabel="Ora")
    plot_cf(f"../plots/{island}/wind_cf_hourly.png", wind_avg, axis=hours_in_a_day, xlabel="Ora")
    plot_cf(f"../plots/{island}/wec_cf_hourly.png", wec_avg, axis=hours_in_a_day, xlabel="Ora")

def graph_monthly_cf(island: str, frame: dict):
    solar_weights = [[] for _ in range(12)]
    wec_weights = [[] for _ in range(12)]
    wind_weights = [[] for _ in range(12)]

    for i, date in enumerate(frame["data"]):
        # month from iso string
        month = int(date.split("-")[1])
        solar_weights[month - 1].append(frame["cf_fotovoltaico"][i])
        wec_weights[month - 1].append(frame["cf_wec"][i])
        wind_weights[month - 1].append(frame["cf_eolico"][i])

    solar_avg = [sum(x) / len(x) for x in solar_weights]
    wec_avg = [sum(x) / len(x) for x in wec_weights]
    wind_avg = [sum(x) / len(x) for x in wind_weights]

    months = [f"{x:02}" for x in range(1, 13)]
    if island == "lipari":
        write_dataframe("../results/lipari_solar_monthly", {
            "month": months,
            "cf": solar_avg,
        })

    plot_cf(f"../plots/{island}/solar_cf_monthly.png", solar_avg, axis=months, xlabel="Mese")
    plot_cf(f"../plots/{island}/wind_cf_monthly.png", wind_avg, axis=months, xlabel="Mese")
    plot_cf(f"../plots/{island}/wec_cf_monthly.png", wec_avg, axis=months, xlabel="Mese")

def make_graphs():
    print("Generating cf...")
    turbine = WindTurbine(
        rotor_diameter=155,
        cut_in_speed=4,
        cut_out_speed=25,
        rated_speed=12,
        rated_power=6_000_000,  # 6 MW
        cp_file="../data/6mw_turbine.csv"
    )

    wec_matrix = WecPowerMatrix("../data/wec_matrix.csv")

    wallis_cf = get_island_cf_and_consumption(
        island_folder="wallis",
        yearly_consumption=ENERGY_CONSUMED_YEARLY_WALLIS,
        turbine=turbine,
        wec_matrix=wec_matrix
    )

    lipari_cf = get_island_cf_and_consumption(
        island_folder="lipari",
        yearly_consumption=ENERGY_CONSUMED_YEARLY_LIPARI,
        turbine=turbine,
        wec_matrix=wec_matrix
    )

    print("Graphing...")

    graph_daily_cf("lipari", lipari_cf)
    graph_monthly_cf("lipari", lipari_cf)
    
    graph_daily_cf("wallis", wallis_cf)
    graph_monthly_cf("wallis", wallis_cf)

    print("Done!")


def change_format():
    frame = read_csv("../data/normalized_power_consumption.csv")
    dates = []
    for date in frame["Time"]:
        print(date)
        dates.append(date.strftime("%Y-%m-%d %H:%M"))

    write_dataframe("../results/energy_consumption.csv", {
        "time": dates,
        "power": frame["Normalized Power Consumption"]
    })


def plot_balancing_data(island: str):
    print(f"Plotting balancing for {island}...")
    print("Reading data...")
    balancing = read_csv(f"../data/{island}/balancing.csv")
    
    bins = {
        "month": [1,2,3,4,5,6,7,8,9,10,11,12],
        "solar": [0] * 12,
        "wind": [0] * 12,
        "wec": [0] * 12,
        "accumulator": [0] * 12,
        "diesel": [0] * 12,
        "used": [0] * 12,
    }

    print("Making Bins...")
    for i in range(len(balancing["date"])):
        bin = balancing["date"][i].month - 1

        used = balancing["energy_used"][i]
        solar = balancing["solar_used"][i]
        wind = balancing["wind_used"][i]
        wec = balancing["wec_used"][i]
        accumulator = balancing["accumulator_used"][i]
        wasted = balancing["wasted_energy"][i]
        diesel = balancing["diesel_used"][i]

        # print(i, used, solar, wind, wec, accumulator, wasted, diesel, error)

        renewable = solar + wind + wec

        if renewable != 0:
            solar_prop = solar / renewable
            wind_prop = wind / renewable
            wec_prop = wec / renewable

            solar -= wasted * solar_prop
            wind -= wasted * wind_prop
            wec -= wasted * wec_prop
        
        error = solar + wind + wec - wasted + accumulator + diesel  - used
        if abs(error) > 10:
            pass
            # print(i, error, abs(accumulator))
            # print(f"{i=}, {used=}, {accumulator=}, {diesel=}, {solar=}, {wind=}, {wec=}, {wasted=}, {error=}")


        bins["solar"][bin] += solar
        bins["wec"][bin] += wec
        bins["wind"][bin] += wind
        bins["accumulator"][bin] += accumulator
        bins["diesel"][bin] += diesel
        bins["used"][bin] += used

    for month in range(len(bins["used"])):
        used = bins["used"][month]
        solar = bins["solar"][month]
        wec = bins["solar"][month]
        wind = bins["solar"][month]
        accumulator = bins["solar"][month]
        diesel = bins["diesel"][month]

    print("Writing bins...")
    write_dataframe(f"../results/{island}_consumption.csv", bins)


def main():
    plot_balancing_data("wallis")
    plot_balancing_data("lipari")
    # make_graphs()
    # lipari_analysis()
    # generate_cf_files()
    return

    turbine = WindTurbine(
        rotor_diameter=155,
        cut_in_speed=4,
        cut_out_speed=25,
        rated_speed=12,
        rated_power=6_000_000,  # 6 MW
        cp_file="../data/6mw_turbine.csv"
    )

    wind = np.linspace(0, 30, 1000)
    # cp = turbine.get_cp(wind)
    energy = [turbine.power_generated(w) for w in wind]

    with open("../plot_data/wind_energy.csv", "w") as tfile:
        tfile.write("wind,energy\n")
        for w, e in zip(wind, energy):
            tfile.write(f"{w},{e}\n")

    plot_data(wind, energy)
    input()

    # make_graphs()


def lipari_analysis():
    # turbine = WindTurbine(
    #     rotor_diameter=126,
    #     cut_in_speed=3,
    #     cut_out_speed=25,
    #     rated_speed=11.4,
    #     rated_power=5_000_000,  # 5 MW
    #     cp_file="../data/turbine_cp.csv"
    # )

    # land_turbine = WindTurbine(
    #     rotor_diameter=150,
    #     cut_in_speed=3.25,
    #     cut_out_speed=25,
    #     rated_speed=9.75,
    #     rated_power=4_000_000,
    #     cp_file="../data/land_turbine.csv"
    # )

    # offshore_turbine = WindTurbine(
    #     rotor_diameter=155,
    #     cut_in_speed=4,
    #     cut_out_speed=25,
    #     rated_speed=12,
    #     rated_power=6_000_000,  # 6 MW
    #     cp_file="../data/6mw_turbine.csv"
    # )

    # turbine = land_turbine

    # wec_matrix = WecPowerMatrix("../data/wec_matrix.csv")

    # print("Calculating for Lipari...")
    # lipari_cf = get_island_cf_and_consumption(
    #     island_folder="lipari",
    #     yearly_consumption=ENERGY_CONSUMED_YEARLY_LIPARI,
    #     turbine=turbine,
    #     wec_matrix=wec_matrix
    # )

    # n = len(lipari_cf["cf_fotovoltaico"])
    # average_wind_cf = sum(lipari_cf["cf_eolico"]) / n
    # average_solar_cf = sum(lipari_cf["cf_fotovoltaico"]) / n
    # average_wec_cf = sum(lipari_cf["cf_wec"]) / n


    wind_frame = read_csv(f"../data/lipari/wind_and_waves_fixed_old.csv")

    time = wind_frame["time"]

    wind_u = np.array(wind_frame["wind_100m_u"])
    wind_v = np.array(wind_frame["wind_100m_v"])

    wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)

    print("Plotting")
    # plot_data_time(time, wind_speed)
    print(np.average(wind_speed))



    # print(f"Average Wind CF: {average_wind_cf * 100:.2f}%")
    # print(f"Average Solar CF: {average_solar_cf * 100:.2f}%")
    # print(f"Average WEC CF: {average_wec_cf * 100:.2f}%")
    # print()




if __name__ == "__main__":
    main()
