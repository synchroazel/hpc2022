import matplotlib.pyplot as plt
import pandas as pd
import math

df = pd.read_csv("data.tsv", delimiter="\t")

time = df["time (sec)"].astype(float)

plt.plot(df["n"], [math.log(t) for t in time])
# plt.plot(df['size'], time)
plt.title("Avg. trip time over buffer size")
plt.xlabel("buffer size")
plt.ylabel("avg. trip time (sec)")
plt.show()

rate = df["rate (MB/sec)"].astype(float)

plt.plot(df["n"], [math.log(r) for r in rate])
# plt.plot(df["n"], rate)
plt.title("Avg. rate over buffer size")
plt.ylabel("avg. rate (MB/sec)")
plt.xlabel("buffer size")
plt.show()