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


def linearRegression(file: str, value: int):
    # Import the data
    data = pd.read_csv(f"{file}.csv")
    print("Data Imported... ")


    # Convert relevant pandas crap to numpy arrays 
    time = data["time"].values.reshape(-1, 1)
    pos = data["position"].values.reshape(-1, 1)
    vel = data["velocity"].values.reshape(-1, 1)
    time.flatten()
    if 0 == value:
        val = pos
    else:
        val = vel
    #print(list(walk_time.flatten()))
    linreg = LinearRegression()
    lin_model = linreg.fit(time, val)
    fit = linreg.predict(time)
    print(type(fit))
    print(f"f(x)={linreg.coef_}x+{linreg.intercept_}")
    chi2 = chiSquare(list(pos.flatten()), list(fit.flatten()))
    #lin_dof = len(time) - 2

    # Create a plot of the data
    plt.scatter(
        time,
        val,
        color="dodgerblue",
        label="collected data"
    )
    plt.plot(
        time,
        fit,
        color='red',
        label="Regression Line"
    )
    plt.plot(
        [],
        [],
        label = f"Chi-Square/DOF = {chi2} / 3"
    )
    print(f"Deviation from Model for Linear Model: {deviationFromModel(pos.flatten(), fit.flatten())}")
    plt.xlabel("Time (sec)")
    if 0 == value:
        plt.ylabel("Position (meters)")
        plt.title("Position over Time: m_2 = 100g")
    else:
        plt.ylabel("Velocity (meters/sec)")
        plt.title("Velocity over Time: m_2 = 100g")
    plt.legend()
    plt.show()


def quadraticRegression(file: str):
    # import the running data
    data = pd.read_csv(f"{file}.csv")
    print("Importing Running Data... ")

    # do a regression
    time = data["Time (s) Run #1"]
    pos = data["Position (m) Run #1"]
    vel = data["Velocity (m/s) Run #1"]
    #accel = running["Acceleration (m/s²) Run #1"]

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


def lab_07_final_vel(
    m1: int,
    m2: int,
    d: float,
    v_o: float
):
    """Finds final velocity using work energy thm"""
    g=-9.81
    v_f = np.sqrt(2*d*g*m2/(m1+m2) + v_o**2)
    print(v_f)
    return v_f

linearRegression("150-grams", 0)
lab_07_final_vel(
    199,
    150,
    -0.35,
    -0.04
)
