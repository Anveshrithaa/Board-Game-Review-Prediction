#A Board game review prediction code that uses Machine Learning algorithms(Random Forest Regression) to predict the average rating of a board game from various Independent variables. The dataset contains info of more than 80,000 board games

import pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
games = pandas.read_csv("games.csv")
# Print the names of the columns in games.
print(games.columns)
print(games.shape)

# Make a histogram of all the ratings in the average_rating column.
plt.hist(games["average_rating"])

# Show the plot.
plt.show()

#There are rows in the dataset that donot have user ratings at all
# Remove the rows without user reviews.
games = games[games["users_rated"] > 0]
# Remove the rows with missing values.
games = games.dropna(axis=0)

# Make a histogram of all the ratings in the average_rating column.
plt.hist(games["average_rating"])

# Show the plot.
plt.show()

#correlation matrix to find correlation between the Independent variables
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

# Get all the columns from the dataframe.
columns = games.columns.tolist()
#Independent variables columns.
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on(dependent variable)
target = "average_rating"

# Splitting the data into Training set and Test set
train = games.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)


# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor

# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error
# Compute the error
mean_squared_error(predictions, test[target])

