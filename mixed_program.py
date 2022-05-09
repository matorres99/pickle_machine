import kfold_template
import pandas
from sklearn import linear_model

dataset = pandas.read_csv("historical_data.csv")

target = dataset.iloc[:,0].values
data = dataset.iloc[:,1:4].values

print(target)
print(data)

results = kfold_template.run_kfold(data, target, 4, linear_model.LinearRegression())
print(results)

machine = linear_model.LinearRegression()
machine.fit(data, target)

new_data = pandas.read_csv("new_data.csv")
print(machine.predict(new_data))