import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.tsv", delimiter="\t")

plt.plot(df["n"], df["time (sec)"])
plt.title("Avg. trip time over buffer size")
plt.xlabel("buffer size")
plt.ylabel("avg. trip time (sec)")
plt.show()

plt.plot(df["n"], df["rate (MB/sec)"])
plt.title("Avg. rate over buffer size")
plt.ylabel("avg. rate (MB/sec)")
plt.xlabel("buffer size")
plt.show()