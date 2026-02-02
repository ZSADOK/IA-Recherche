import matplotlib.pyplot as plt
import csv

t, mse = [], []
with open("./anytime.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        t.append(float(row["time"]))
        mse.append(float(row["mse"]))

plt.plot(t, mse)
plt.xlabel("Time (s)")
plt.ylabel("MSE")
plt.title("Anytime behavior")
plt.show()
