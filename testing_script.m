histfit(data.X_train(:,306))
histfit(data.X_train(:,524))

data = FeatureProcess.StdScale(data)
histfit(data.X_train(:,306))
histfit(data.X_train(:,524))