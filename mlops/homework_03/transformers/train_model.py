categorical = ['PULocationID', 'DOLocationID']
features = df[categorical]
# we convert the data type fron integer to string before passing it to the dictionary vectorizer
df[categorical] = df[categorical].astype(str) 

# next we convert the categorical variables to a dictionary
train_dicts = df[categorical].to_dict(orient = 'records')

# create an instance of the dictionary vectorizer
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

y_train = df['duration']

# Linear regression model
lin_reg = LinearRegression()

# fit data to model
lin_reg.fit(X_train, y_train)

y_train_pred = lin_reg.predict(X_train)