from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from collections import deque
from scipy import optimize
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random

def transform_trading_days_to_trading_weeks(df):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        classification =  df_trading_week.iloc[0][['Classification']].values[0]
        opening_day_of_week = df_trading_week.iloc[0][['Open']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], opening_day_of_week, closing_day_of_week, classification])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Open', 'Week Close', 'Classification'])
    return trading_list_df

def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels
    '''
    # The weekly balance we will be using
    weekly_balance_acc = weekly_balance
    trading_history = deque()
    index = 0
    
    while(index < len(trading_df.index) - 1):
        trading_week_index = index
        if weekly_balance_acc != 0:
            # Find the next consecutive green set of weeks and trade on them
            while(trading_week_index < len(trading_df.index) - 1 and trading_df.iloc[trading_week_index][[prediction_label]].values[0] == 'GREEN'):
                trading_week_index += 1
            green_weeks = trading_df.iloc[index:trading_week_index][['Week Open', 'Week Close']]
            # Check if there are green weeks, and if there are not, we add a row for trading history
            if len(green_weeks.index) > 0:
                # Buy shares at open and sell shares at close of week
                green_weeks_open = float(green_weeks.iloc[0][['Week Open']].values[0])
                green_weeks_close = float(green_weeks.iloc[-1][['Week Close']].values[0])
                # We append the money after we make the trade
                weekly_balance_acc = make_trade(weekly_balance_acc, green_weeks_open, green_weeks_close)
            # Regardless of whether we made a trade or not, we append the weekly cash and week over
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                weekly_balance_acc])
        else:
            # If we have no money we will not be able to trade
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                    trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                    weekly_balance_acc])
        index = trading_week_index+1
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Trading Week', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df


def main():
    ticker='WMT'
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'
    file_name = 'WMT_weekly_return_volatility.csv'
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]

    fit_x_training = df_2018[['mean_return', 'volatility']]
    fit_y_training = df_2018[['Classification']].values
    fix_x_test = df_2019[['mean_return', 'volatility']]
    fit_y_test = df_2019[['Classification']].values

    print('The selected combination may yield different results because it\'s a random selection from a set of optimal N and D values.')
    print('\nQuestion 1:')

    error_random_tree = np.zeros((10, 5))
    error_random_tree_x = deque()

    for n in range(1, 11):
        for d in range(1,6):
            fit_random_tree_classifier = RandomForestClassifier(max_depth=d, n_estimators=n, criterion='entropy').fit(fit_x_training, fit_y_training.ravel())
            y_pred_random_tree = fit_random_tree_classifier.predict(fix_x_test)
            error_random_tree[n-1][d-1] = np.round(100 - np.multiply(np.mean(y_pred_random_tree == fit_y_test), 100), 2)
            error_random_tree_x.append('{}, {}'.format(n, d))
    min_indices = np.argwhere(error_random_tree == np.min(error_random_tree.flatten()))
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 8)
    ax1.plot(np.array(error_random_tree_x), error_random_tree.flatten())
    ax1.set_xlabel('N, D')
    ax1.set_ylabel('Error (%)')
    plt.savefig(fname='Q_1_N_D_vs_Error')
    plt.xticks(rotation=90)
    plt.title('Error Random Tree')
    plt.show()
    plt.close()

    print('Minimum error rate: {}%'.format(np.min(error_random_tree.flatten())))
    print('The following N, D values have the minimum errors: \n{}'.format(min_indices + 1))
    random_index = random.randrange(0, len(min_indices)-1)
    selected_n_d = min_indices[random_index] + 1
    print('There are most likely multiple N, D pairs with the same minimum error, so I will be using a random set from them.')
    print('I will use the random set: n={}, d={})'.format(selected_n_d[0], selected_n_d[1]))

    print('\nQuestion 2:')
    print('Random Forest Confusion matrix')
    fit_random_tree_classifier = RandomForestClassifier(max_depth=selected_n_d[0], n_estimators=selected_n_d[1], criterion='entropy').fit(fit_x_training, fit_y_training.ravel())
    y_pred_random_tree = fit_random_tree_classifier.predict(fix_x_test)
    confusion_matrix_array_random_tree = confusion_matrix(fit_y_test, y_pred_random_tree)
    confusion_matrix_df_random_tree = pd.DataFrame(confusion_matrix_array_random_tree, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df_random_tree)

    print('\nQuestion 3:')
    total_data_points = len(fit_y_test)
    true_positive_number = confusion_matrix_df_random_tree['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df_random_tree['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print('True positive rate: {}%'.format(true_positive_rate))
    print('True negative rate: {}%'.format(true_negative_rate))

    print('\nQuestion 4:')
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')
    
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels Random Forest", y_pred_random_tree, allow_duplicates=True)
    balance_end_random_forest = trading_strategy(trading_weeks_2019, "Predicted Labels Random Forest")[['Balance']].values[-1][0]
    balance_end_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")[['Balance']].values[-1][0]

    print('With Buy and Hold: ${}'.format(balance_end_hold))
    print('With Random Forest: ${}'.format(balance_end_random_forest))
    if balance_end_random_forest > balance_end_hold:
        print('Random Forest is a better strategy.')
    else:
        print('Buy and hold is a better strategy.')
    
if __name__ == "__main__":
    main()