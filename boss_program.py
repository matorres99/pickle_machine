#we call this boss program because this is the file that our boss would run, it has the machine already trained
#and saved via the pickle package, does not need the historical data. all that is needed is this program.

import pandas
import pickle

with open("machine.pickle", "rb") as f:
	machine = pickle.load(f)


new_data = pandas.read_csv("new_data.csv")
print(machine.predict(new_data))