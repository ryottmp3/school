import pandas as pd

c = 2.9988e8
h = 6.626e-34

def Energy(w):
    def Freq(w):
        nu = c / w
        return nu

    nu = Freq(w)

    E = h * nu

    return E, nu


if __name__ == '__main__':
    df = pd.read_csv('colors.csv', header=0)

    df['Energy'], df['Frequency'] = Energy(df['wavelength'])

    print(df)
