from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from collections import deque
from scipy import optimize
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from wang_assignment_9_random_forest import trading_strategy, make_trade, transform_trading_days_to_trading_weeks
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as plt

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

    print('\nQuestion 1:')
    print('RF is Random Forest, DT is Decision Tree, GNB is Gaussian Naive Bayes')
    lambda_values = [0.5, 1]
    lambda_1_error = np.zeros((15, 3))
    lambda_1_x = deque()
    lambda_2_error = np.zeros((15, 3))
    lambda_2_x = deque()

    for rate in lambda_values:
        for n in range(1, 16):
            random_forest_classifier = RandomForestClassifier(n_estimators=n, criterion='entropy')
            decision_tree_classifier = tree.DecisionTreeClassifier()
            gaussian_nb = GaussianNB()
            classifier_arr = [random_forest_classifier, decision_tree_classifier, gaussian_nb]
            classifier_arr_name = ['RF', 'DT', 'GNB']
            for index, base_estimator in enumerate(classifier_arr):
                adaboost_classifier = AdaBoostClassifier(n_estimators=n, base_estimator=base_estimator, learning_rate=rate)
                fit_adaboost_classifier= adaboost_classifier.fit(fit_x_training, fit_y_training.ravel())

                y_pred_decision_tree = fit_adaboost_classifier.predict(fix_x_test)
                error =  np.round(100 - np.multiply(np.mean(y_pred_decision_tree == fit_y_test), 100), 2)
                if rate == 0.5:
                    lambda_1_error[n-1][index] = error
                    lambda_1_x.append('{}, {}'.format(n, classifier_arr_name[index]))
                else:
                    lambda_2_error[n-1][index] = error
                    lambda_2_x.append('{}, {}'.format(n, classifier_arr_name[index]))

    min_indices = np.argwhere(lambda_1_error == np.min(lambda_1_error.flatten()))

    fig, ax1 = plt.subplots()
    plt.xticks(rotation=90)
    plt.title('Error Adaboost')
    color = 'tab:blue'
    ax1.set_xlabel('N, Classifier')
    ax1.set_ylabel('Error (%)')
    fig.set_size_inches(13, 9)
    plot_1 = ax1.plot(np.array(lambda_1_x), lambda_1_error.flatten(), color=color, label='Rate=0.5')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    plot_2 = ax2.plot(np.array(lambda_2_x), lambda_2_error.flatten(), color=color, label='Rate=1')
    ax2.set_xlabel('N, Classifier')
    ax2.set_ylabel('Error (%)')

    lns = plot_1+plot_2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.show()
    plt.close()

    print('\nQuestion 2:')
    print('Minimum error rate lambda = 0.5: {}%'.format(np.min(lambda_1_error.flatten())))
    print('Minimum error rate lambda = 1: {}%'.format(np.min(lambda_2_error.flatten())))
    

    # print('Accuracy for year 2 Decision Tree: {}%'.format(accuracy_decision_tree))

    # print('Decision Tree Confusion matrix')
    # confusion_matrix_array_decision_tree = confusion_matrix(fit_y_test, y_pred_decision_tree)
    # confusion_matrix_df_decision_tree = pd.DataFrame(confusion_matrix_array_decision_tree, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    # print(confusion_matrix_df_decision_tree)

    # print('\nQuestion 3:')
    # total_data_points = len(fit_y_test)
    # true_positive_number = confusion_matrix_df_decision_tree['Predicted: GREEN']['Actual: GREEN']
    # true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    # true_negative_number = confusion_matrix_df_decision_tree['Predicted: RED']['Actual: RED']
    # true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    # print('True positive rate: {}%'.format(true_positive_rate))
    # print('True negative rate: {}%'.format(true_negative_rate))

    # print('\nQuestion 4:')
    # df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    # df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    # trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    # buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')

    # trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    # trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels Decision Tree", y_pred_decision_tree, allow_duplicates=True)
    # balance_end_decision_tree = trading_strategy(trading_weeks_2019, "Predicted Labels Decision Tree")[['Balance']].values[-1][0]
    # balance_end_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")[['Balance']].values[-1][0]

    # print('With buy and hold: ${}'.format(balance_end_hold))
    # print('With Decision Tree: ${}'.format(balance_end_decision_tree))
    # print('Buy and hold is a better strategy.')
    
if __name__ == "__main__":
    main()