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

def plot_adaboost(lambda_1_x, lambda_2_x, lambda_1_error, lambda_2_error, classifier_name):
    fig, ax1 = plt.subplots()
    plt.xticks(rotation=90)
    plt.title('Error Adaboost {}'.format(classifier_name))
    color = 'tab:blue'
    ax1.set_xlabel('N, Classifier')
    ax1.set_ylabel('Error (%)')
    fig.set_size_inches(13, 9)
    plot_1 = ax1.plot(lambda_1_x, lambda_1_error, color=color, label='Rate=0.5')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    plot_2 = ax2.plot(lambda_2_x, lambda_2_error, color=color, label='Rate=1')
    ax2.set_xlabel('N, Classifier')
    ax2.set_ylabel('Error (%)')

    lns = plot_1+plot_2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig(fname='Adaboost_{}'.format(classifier_name))
    plt.show()
    plt.close()


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

    print('Depending on the strategy found, the best strategy is used for trading. The results may vary on different runs.')

    print('\nQuestion 1:')
    print('RF is Random Forest, DT is Decision Tree, GNB is Gaussian Naive Bayes')
    lambda_values = [0.5, 1]
    lambda_1_error = np.zeros((15, 3))
    lambda_1_x = np.empty((15, 3), dtype="S10")
    lambda_2_error = np.zeros((15, 3))
    lambda_2_x = np.empty((15, 3), dtype="S10")

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

                y_pred = fit_adaboost_classifier.predict(fix_x_test)
                error =  np.round(100 - np.multiply(np.mean(y_pred == fit_y_test), 100), 2)
                if rate == 0.5:
                    lambda_1_error[n-1][index] = error
                    lambda_1_x[n-1][index] = '{}, {}'.format(n, classifier_arr_name[index])
                else:
                    lambda_2_error[n-1][index] = error
                    lambda_2_x[n-1][index] = '{}, {}'.format(n, classifier_arr_name[index])

    plot_adaboost(lambda_1_x.T[0], lambda_2_x.T[0], lambda_1_error.T[0], lambda_2_error.T[0], 'Random_Forest')
    plot_adaboost(lambda_1_x.T[1], lambda_2_x.T[1], lambda_1_error.T[1], lambda_2_error.T[1], 'Decision_Tree')
    plot_adaboost(lambda_1_x.T[2], lambda_2_x.T[2], lambda_1_error.T[2], lambda_2_error.T[2], 'Gaussian_Naive_Bayes')



    print('\nQuestion 2:')
    print('Minimum error rate lambda = 0.5: {}%'.format(np.min(lambda_1_error.flatten())))
    print('Minimum error rate lambda = 1: {}%'.format(np.min(lambda_2_error.flatten())))
    
    print('Best value(s) of N for each base estimator: ')
    error_by_classifier = lambda_1_error.T

    # Lowest Errors
    min_indices_random_forest = np.argwhere(error_by_classifier[0] == np.min(error_by_classifier[0]))
    min_indices_decision_tree = np.argwhere(error_by_classifier[1] == np.min(error_by_classifier[1]))
    min_indices_naive_bayes = np.argwhere(error_by_classifier[2] == np.min(error_by_classifier[2]))

    # Taking the first min if there are multiple indices
    if len(min_indices_random_forest) > 0:
        min_indices_random_forest = min_indices_random_forest[0]
    if len(min_indices_decision_tree) > 0:
        min_indices_decision_tree = min_indices_decision_tree[0]
    if len(min_indices_naive_bayes) > 0:
        min_indices_naive_bayes = min_indices_naive_bayes[0]

    print('Random Forest: N*={}'.format((min_indices_random_forest + 1).flatten()[0]))
    print('Decision Tree: N*={}'.format((min_indices_decision_tree + 1).flatten()[0]))
    print('Naive Bayes: N*={}'.format((min_indices_naive_bayes + 1).flatten()[0]))

    print('\nQuestion 3:')
    print('Accuracy for best base estimator')
    min_random_forest = (min_indices_random_forest + 1).flatten()[0]
    min_decision_tree = (min_indices_decision_tree + 1).flatten()[0]
    min_naive_bayes = (min_indices_naive_bayes + 1).flatten()[0]

    accuracy_random_forest = (100 - error_by_classifier[0][min_indices_random_forest]).flatten()[0]
    accuracy_decision_tree = (100 - error_by_classifier[0][min_indices_decision_tree]).flatten()[0]
    accuracy_naive_bayes = (100 - error_by_classifier[0][min_indices_naive_bayes]).flatten()[0]
    print('Accuracy Random Forest: {}%'.format(accuracy_random_forest))
    print('Accuracy Decision Tree: {}%'.format(accuracy_decision_tree))
    print('Accuracy Naive Bayes: {}%'.format(accuracy_naive_bayes))

    print('\nQuestion 4:')
    accuracy_classifications = [accuracy_random_forest, accuracy_decision_tree, accuracy_naive_bayes]
    max_arg_classifications = np.argmax(accuracy_classifications)
    if max_arg_classifications == 0:
        print('Random Forest is the best base estimator')
        best_estimator = 'Random Forest'
    if max_arg_classifications == 1:
        print('Decision Tree is the best base estimator')
        best_estimator = 'Decision Tree'
    if max_arg_classifications == 2:
        print('Naive Bayes is the best base estimator')
        best_estimator = 'Naive Bayes'
    print('If two estimators are tied, they are both considered the best estimator and we will use the first of the two.')

    print('\nQuestion 5:')
    if max_arg_classifications == 0:
        # Random Forest
        n = (min_indices_random_forest + 1).flatten()[0]
        random_forest_classifier = RandomForestClassifier(n_estimators=n, criterion='entropy')
        adaboost_classifier = AdaBoostClassifier(n_estimators=n, base_estimator=random_forest_classifier, learning_rate=0.5)
        fit_adaboost_classifier= adaboost_classifier.fit(fit_x_training, fit_y_training.ravel())
    elif max_arg_classifications == 1:
        # Decision Tree
        n = (min_indices_decision_tree + 1).flatten()[0]
        decision_tree_classifier = tree.DecisionTreeClassifier()
        adaboost_classifier = AdaBoostClassifier(n_estimators=n, base_estimator=decision_tree_classifier, learning_rate=0.5)
        fit_adaboost_classifier= adaboost_classifier.fit(fit_x_training, fit_y_training.ravel())
    elif max_arg_classifications == 2:
        # Naive bayes
        n = (min_indices_naive_bayes + 1).flatten()[0]
        gaussian_nb = GaussianNB()
        adaboost_classifier = AdaBoostClassifier(n_estimators=n, base_estimator=gaussian_nb, learning_rate=0.5)
        fit_adaboost_classifier= adaboost_classifier.fit(fit_x_training, fit_y_training.ravel())
    # Fit X test
    y_pred = fit_adaboost_classifier.predict(fix_x_test)

    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')

    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels Best Estimator", y_pred, allow_duplicates=True)
    balance_end_strat = trading_strategy(trading_weeks_2019, "Predicted Labels Best Estimator")[['Balance']].values[-1][0]
    balance_end_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")[['Balance']].values[-1][0]

    print('With buy and hold: ${}'.format(balance_end_hold))
    print('With {}: ${}'.format(best_estimator, balance_end_strat))
    if balance_end_strat > balance_end_hold:
        print('Random Forest is a better strategy.')
    else:
        print('Buy and hold is a better strategy.')
    
if __name__ == "__main__":
    main()