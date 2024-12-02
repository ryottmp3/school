# fit.py: fitting data for phys lab
# copyright © 2024 H. Ryott Glayzer
# MIT License
import pandas as pd
import matplotlib as mpl
mpl.use("gtk4agg")
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression

def chiSquare(observed, expected):
    """Performs a chi squared goodness-of-fit test"""
    middleVal = []
    for i in range(len(observed)):
        value = ((observed[i] - expected[i])**2 / expected[i])
        middleVal.append(value)

    chiSquareStatistic = sum(middleVal)
    return chiSquareStatistic

def deviationFromModel(obs, exp):
    partSum = []
    for i in range(len(obs)):
        ama = (obs[i] - exp[i])**2
        partSum.append(ama)
    wholeSum = sum(partSum)
    deviation = np.sqrt(wholeSum / len(obs))

    return deviation

def main():
    # Import the data
    walking = pd.read_csv("walking.csv")
    print("Walking Data Imported... ")

    # Make a cut of the first walk away
    walking = walking.head(146)
    print("Making time cuts in Walking Data... ")

    # Convert relevant pandas crap to numpy arrays 
    walk_time = walking["Time (s) Run #1"].values.reshape(-1, 1)
    walk_pos = walking["Position (m) Run #1"].values.reshape(-1, 1)
    walk_time.flatten()
    #print(list(walk_time.flatten()))
    linreg = LinearRegression()
    lin_model = linreg.fit(walk_time, walk_pos)
    fit = linreg.predict(walk_time)
    print(type(fit))
    print(f"f(x)={linreg.coef_}x+{linreg.intercept_}")
    walk_chi2 = chiSquare(list(walk_pos.flatten()), list(fit.flatten()))
    walk_dof = len(walk_time) - 2

    # Create a plot of the data
    plt.scatter(
        walk_time,
        walk_pos,
        color="dodgerblue",
        label="collected data"
    )
    plt.plot(
        walk_time,
        fit,
        color='red',
        label="Regression Line"
    )
    plt.plot(
        [],
        [],
        label = f"Chi-Square/DOF = {walk_chi2} / 3"
    )
    print(f"Deviation from Model for Walking Model: {deviationFromModel(walk_pos.flatten(), fit.flatten())}")
    plt.xlabel("Time (sec)")
    plt.ylabel("Position (meters)")
    plt.title("Position vs Time of a Student Walking")
    plt.legend()
    plt.show()

    # import the running data
    running = pd.read_csv("running.csv")
    print("Importing Running Data... ")

    # Make a quality cut
    running = running.head(46)
    print("Making Time Cuts in Running Data")

    # do a regression
    run_time = running["Time (s) Run #1"]
    run_pos = running["Position (m) Run #1"]
    run_vel = running["Velocity (m/s) Run #1"]
    run_accel = running["Acceleration (m/s²) Run #1"]

    quad_model = np.poly1d(np.polyfit(run_time, run_pos, 2))
    print(quad_model)

    run_chi2 = chiSquare(run_pos, quad_model(run_time))
    run_dof = len(run_time) - 2

    # Create a plot of the data
    plt.scatter(
        run_time,
        run_pos,
        color='dodgerblue',
        label="Collected Data"
    )
    plt.plot(
        run_time,
        quad_model(run_time),
        color='red',
        label="Quadratic Model"
    )
    plt.plot(
        [],
        [],
        label = f"Chi-Square / DOF : {run_chi2} / 3"
    )
    print(f"Deviation from Model for Quadratic Fit: {deviationFromModel(run_pos, quad_model(run_time))}")
    plt.title("Kinematics of a Running Student")
    plt.xlabel("Time (sec)")
    plt.ylabel("Position (meters)")
    plt.legend()
    plt.show()

    # Plot measured instantaneous acceleration vs time
    plt.scatter(
        run_time,
        run_accel,
        color="green",
        label="Instantaneous Acceleration"
    )
    plt.plot(
        run_time,
        run_accel,
        color="green"
    )
    plt.scatter(
        run_time,
        run_vel,
        color="orange",
        label="Instantaneous Velocity"
    )
    plt.plot(
        run_time,
        run_vel,
        color="orange"
    )
    plt.scatter(
        run_time,
        run_pos,
        color="dodgerblue",
        label="Instantaneous Position"
    )
    plt.plot(
        run_time,
        run_pos,
        color="dodgerblue"
    )
    plt.title("Instantaneous Kinematics of a Running Student")
    plt.xlabel("Time (sec)")
    plt.ylabel("Acceleration (m/s^2), Velocity (m/s), and Position (m)")
    plt.legend()
    plt.show()

    # Find the average acceleration using the collected data
    avg_accel = running.mean(axis=0)
    print(f"Mean Acceleration: {avg_accel}")

main()
