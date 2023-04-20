import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, MonthLocator
from utils import read_csv


def main():
    df = read_csv("../data/wallis_north_wind_and_waves.csv", ignore=["index"])

    print("Creating Plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    x_data = df["time"]
    y_data = df["significant_wave_height"]

    ax.plot(x_data, y_data)

    # Tick Every Month
    ax.xaxis.set_major_locator(MonthLocator())

    # Show date as month abbreviation
    ax.xaxis.set_major_formatter(DateFormatter("%b"))

    # Add blue below data
    ax.fill_between(x_data, 0, y_data, alpha=.3)

    # Show a grid
    ax.grid()

    ax.autoscale(enable=True, axis="both", tight=True)
    fig.savefig("../plots/wave_height.png")

    print("Done!")


if __name__ == "__main__":
    main()
