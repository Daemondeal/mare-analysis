from utils import read_csv
import matplotlib.pyplot as plt
import numpy as np


def main():
    frame = read_csv("../data/wallis/balancing.csv")

    graph_daily_cf("wallis", frame)


def plot_balancing(filename: str, cf, *, axis, xlabel: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(axis, weights=[x * 100 for x in cf], bins=np.arange(len(cf) + 1)-0.5, linewidth=1.2, edgecolor="black")

    x_line = np.linspace(1, 24, 24)
    y_line = [100 for x in x_line]
    ax.plot(x_line, y_line)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("SOC (%)")
    
    fig.savefig(filename)

def graph_daily_cf(island: str, frame: dict):
    soc_weights = [[] for _ in range(24)]
    
    for i in range(len(frame["charge"]) // 24):
        for j in range(24):
            soc_weights[j].append(frame["charge"][i * 24 + j])

    soc_weights = [sum(x) / (len(x) * 26000000) for x in soc_weights]

    hours_in_a_day = [f"{x:02}" for x in range(24)]

    plot_balancing(f"../plots/{island}/soc.png", soc_weights, axis=hours_in_a_day, xlabel="Ora")

if __name__ == "__main__":
    main()
