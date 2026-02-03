import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("imdb_movies.csv", encoding = "latin1")
print(df.head())
print(df.columns)

df = df[['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Rating']]
df.dropna(inplace=True)

df['Year'] = df['Year'].astype(str).str.replace(r'[^\d]', '',regex=True)
df['Year'] = df['Year'].astype(int)

df['Duration'] = df['Duration'].astype(str).str.replace(' min','')
df['Duration'] = df['Duration'].astype(int)

df['Votes'] = df['Votes'].astype(str).str.replace(',','')
df['Votes'] = df['Votes'].astype(int)

le_genre = LabelEncoder()
le_director = LabelEncoder()
df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
X = df.drop('Rating', axis = 1)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ",mse)