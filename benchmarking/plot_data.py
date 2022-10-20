import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rate and time data")
    parser.add_argument("-p", "--path", type=str, default="data.tsv", help="Path of the .tsv file generated")
    parser.add_argument("-l", "--log-scale", action="store_true", default=False, help="Display data in log scale")

    args = parser.parse_args()

    df = pd.read_csv(args.path, delimiter="\t")

    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(10)

    title = "Trip time and Rate" if not args.log_scale else "Trip time and Rate (log scale)"
    fig.suptitle(title)

    time = df["time (sec)"].astype(float)
    rate = df["rate (MB/sec)"].astype(float)

    if args.log_scale == True:
        axs[0].plot(df["n"], [math.log(t) for t in time])
        axs[1].plot(df["n"], [math.log(r) for r in rate])

    else:
        axs[0].plot(df["n"], time)
        axs[1].plot(df["n"], rate)

    axs[0].set_title("Avg. trip time over buffer size")
    axs[0].set_xlabel("buffer size")
    axs[0].set_ylabel("avg. trip time (sec)")

    axs[1].set_title("Avg. rate over buffer size")
    axs[1].set_ylabel("avg. rate (MB/sec)")
    axs[1].set_xlabel("buffer size")

    plt.show()
