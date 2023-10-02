from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datacleanup import *
import statsmodels.api as sm

def regress(X_train, y_train, X_test, y_test):
    reg = sm.Logit(y_train,X_train).fit()
    #reg = LogisticRegression(C=1, fit_intercept=True)

    #reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print("summary: ", reg.summary())
    #print("coefficients: ", reg.coef_, "intercept: ", reg.intercept_)

    return y_pred

# Function that normalizes features in training set to zero mean and unit variance.
# Input: training data X_train
# Output: the normalized version of the feature matrix: X, the mean of each column in
# training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):
    # fill in
    X = np.zeros((len(X_train[:, 0]), len(X_train[0, :])))
    # print(len(X_train[0, :]))
    # print(len(X_train[:, 0]))
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    #print(X[30][8])

    for i in range(len(X_train[:, 0])):
        for j in range(len(X_train[i, :])):
            X[i][j] = (X_train[i][j] - mean[j]) / std[j]
    return X, mean, std


# Function that normalizes testing set according to mean and std of training set
# Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
# column in training set: trn_std
# Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):
    # fill in

    X = np.zeros((len(X_test[:, 0]), len(X_test[0, :])))

    for i in range(len(X_test[:, 0])):
        for j in range(len(X_test[i, :])):
            X[i][j] = (X_test[i][j] - trn_mean[j]) / trn_std[j]

    return X

if __name__ == '__main__':
    [data, ids] = getFinalData2()

    data = data[:, 1:]#get rid of video ID
    print("Shape: ", data.shape)
    #data = np.delete(data, [0], 1)
    print(data)

    # for training and testing a 90/10 split was used
    bound = round(data.shape[0] * 0.9)
    print("bound: ", bound)
    #test = data[0:bound, 0:data.shape[1]-2]
    #print("test: ", test)

    X_train = data[0:bound, 0:data.shape[1]-1]
    y_train = data[0:bound, data.shape[1] - 1]
    X_test = data[bound:data.shape[0], 0:data.shape[1]-1]
    y_test = data[bound:data.shape[0], data.shape[1] - 1]

    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #print(X_train)
    #print(X_test)
    #print(y_train)

    y_pred = regress(X_train, y_train, X_test, y_test)
    pred = list(map(round, y_pred))
    print("First ten predictions: ", pred[0:10])
    print("First ten values: ", y_test[0:10])
    print("mse: ", mean_squared_error(y_test, y_pred))
    print("r2: ", r2_score(y_test, pred))
    # y_pred = [round(x-0.15) for x in y_pred]
    # print("after round mse: ",mean_squared_error(y_pred,y_test))
    # print("after round r2: ", r2_score(y_pred, y_test))
    # print(y_test)