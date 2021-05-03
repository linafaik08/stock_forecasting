import time
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
import copy

import datetime

from itertools import product
import shutil

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import plotly.figure_factory as ff

pio.templates.default = "none"
pio.renderers.default = 'notebook'

from scipy.optimize import Bounds, LinearConstraint, minimize

from IPython.display import clear_output

from yahoofinancials import YahooFinancials


class Portfolio:
    def __init__(self, initial_portfolio, stickers, verbose=True):
        self.composition = initial_portfolio
        self.stickers = stickers

        if 'liquidity' not in self.composition:
            self.pf_composition['liquidity'] = 0

        self.value = self.get_pf_value()
        self.rate = 0.5 / 100  # bank fees
        self.verbose = verbose

    def get_pf_value(self, composition=None, t=None, forecasts=None, prices=None):
        if composition == None:
            composition = self.composition

        value = composition['liquidity']

        for stock_name, stock_n in composition.items():
            if stock_name == 'liquidity' or stock_n == 0:
                continue
            if t is None:
                stock_price = self.get_stock_price(stock_name) if prices is None else prices[stock_name]
            elif t is not None and forecasts is None:
                stock_price = self.get_stock_price(stock_name, t)
            else:
                stock_price = forecasts[forecasts.ds == t][stock_name].values[0]
            value += stock_price * stock_n
        return value

    def get_fees(self, price_transaction):
        # tarifs Découverte de Boursorma
        if price_transaction <= 398 or price_transaction > 500:
            return self.rate * price_transaction
        else:
            return 1.99

    def get_stock_price(self, stock_name, t=None):
        stock = YahooFinancials(self.stickers[stock_name])
        if t is None:
            price = stock.get_current_price()
        else:
            data = stock.get_historical_price_data(start_date=str(t)[:10],
                                                   end_date=str(datetime.date.today()),
                                                   time_interval='daily')

            data = pd.DataFrame(data[self.stickers[stock_name]]['prices'])
            data.rename(columns={'formatted_date': 'ds', 'close': 'y'}, inplace=True)

            if not (data.empty) and str(t)[:10] in data.ds.unique():
                price = data[data.ds == str(t)[:10]]['y'].values[0]
            else:
                price = np.nan

        return price

    def get_transaction_outcome(self, op_sell={}, op_buy={}, prices=None):

        new_composition = copy.deepcopy(self.composition)
        fees = 0

        for stock_name, n in op_sell.items():

            if stock_name not in self.composition.keys():
                new_composition[stock_name] = 0

            if self.composition[stock_name] < n:
                print('Warning: only {} stocks of {} can be sold (instead of {})'.format(
                    min(new_composition[stock_name], n), stock_name, n))
                print()
                n = min(new_composition[stock_name], n)
            new_composition[stock_name] -= n

            stock_price = self.get_stock_price(stock_name) if prices is None else prices[stock_name]

            new_composition['liquidity'] += stock_price * n - self.get_fees(stock_price * n)

            fees += self.get_fees(stock_price * n)

        for stock_name, n in op_buy.items():

            stock_price = self.get_stock_price(stock_name) if prices is None else prices[stock_name]

            transaction_price = stock_price * n + self.get_fees(stock_price * n)

            if new_composition['liquidity'] < (transaction_price):
                n_new = int((new_composition['liquidity'] - self.get_fees(stock_price * n)) / stock_price)
                n_new = n - 1 if n_new == n else n_new
                print('Warning: only {} stocks of {} can be bought (instead of {})'.format(n_new,
                                                                                           stock_name, n))
                print()
                n = n_new

            if stock_name in new_composition.keys():
                new_composition[stock_name] += n
            else:
                new_composition[stock_name] = n

            new_composition['liquidity'] += -stock_price * n - self.get_fees(stock_price * n)
            fees += self.get_fees(stock_price * n)

            if self.verbose:
                profit_transaction = self.get_pf_value(composition=new_composition, prices=prices) - self.get_pf_value()
                print('Profit before transaction fees: ', round(profit_transaction + fees, 4))
                print('Brokage fees: ', round(fees, 4))
                print('Profit after transaction fees: ', round(profit_transaction, 4))

        return new_composition

    def get_best_operation(self, t, forecasts, prices0=None):
        t0 = time.time()

        alpha0 = [self.composition[stock_name] if stock_name in self.composition.keys() else 0 for stock_name in
                  list(forecasts.columns[1:])]
        if prices0 is None:
            if self.verbose:
                print('----- Getting stock prices')
            prices0 = {stock_name: pf.get_stock_price(stock_name) for stock_name in list(forecasts.columns[1:])}
        list_prices0 = list(prices0.values())

        prices = {s: v for s, v in zip(list(forecasts.columns[1:]),
                                       list(forecasts[forecasts.ds == t][list(forecasts.columns[1:])].values[0]))}
        list_prices = list(prices.values())

        liq0 = self.composition['liquidity']

        def profit(x):
            change_pf = sum([x[i] * list_prices[i] - alpha0[i] * list_prices0[i] for i in range(len(list_prices0))])
            change_liq = sum([(alpha0[i] - x[i]) * list_prices0[i] for i in range(len(list_prices0))])
            return -((1 - self.rate) * change_pf + change_liq)  # because minimization

        # no negative values
        linear_constraint1 = LinearConstraint(np.identity(len(alpha0)),
                                              np.zeros((len(alpha0))),
                                              np.inf * np.ones((len(alpha0))))
        # enough liquidity
        linear_constraint2 = LinearConstraint(np.array(list_prices0), np.array([-np.inf]), np.array([liq0]))

        x0 = np.zeros((len(alpha0)))

        if self.verbose:
            print('----- Optimizing')

        res = minimize(profit, x0, method='trust-constr',
                       constraints=[linear_constraint1, linear_constraint2],
                       options={'verbose': 0})

        op = {list(forecasts.columns[1:])[i]: round(res.x[i]) - alpha0[i] for i in range(len(alpha0))}

        op_buy = {stock_name: change for stock_name, change in op.items() if change > 0}
        op_sell = {stock_name: -change for stock_name, change in op.items() if change < 0}

        new_composition = self.get_transaction_outcome(op_buy=op_buy, op_sell=op_sell, prices=prices0)

        if self.verbose:
            print('----- Computing best operation outcomes')

        outcome = {
            'date_target': t,
            'pf_current_composition': self.composition,
            'pf_current_value': self.get_pf_value(),
            'operation': op,
            'pf_new_composition': new_composition,
            'pf_new_composition_value': self.get_pf_value(composition=new_composition, prices=prices0)
        }

        if self.verbose:
            print('----- Convergence:', res['success'], '({}min)'.format(round((time.time() - t0) / 60)))
            print('----- Best operation:', '\n', op)

        return outcome

    def get_outcome_ROI(self, outcome, forecasts, forecasts_low, forecasts_high, rolling_period=0):

        t = outcome['date_target']
        pf_value = outcome['pf_current_value']
        new_composition = outcome['pf_new_composition']

        if rolling_period == 0:

            for suffix, data in zip(['', '_low', '_high'], [forecasts, forecasts_low, forecasts_high]):
                outcome['pf_value_forecast' + suffix] = self.get_pf_value(composition=new_composition, t=t,
                                                                          forecasts=data)
                outcome['op_profit' + suffix] = outcome['pf_value_forecast' + suffix] - pf_value
                outcome['op_ROI' + suffix] = (outcome['op_profit' + suffix] / pf_value) * 100

        else:

            for suffix, data in zip(['', '_low', '_high'], [forecasts, forecasts_low, forecasts_high]):

                pf_future_values = [
                    self.get_pf_value(composition=new_composition, t=t - datetime.timedelta(i), forecasts=data) for i in
                    range(rolling_period + 1)]
                if self.verbose and suffix == '':
                    print('Portfolio value over the last {} days of the next period:'.format(rolling_period))
                    print(pf_future_values)

                outcome['pf_value_forecast' + suffix] = np.mean(pf_future_values)
                outcome['op_profit' + suffix] = outcome['pf_value_forecast' + suffix] - pf_value
                outcome['op_ROI' + suffix] = (outcome['op_profit' + suffix] / pf_value) * 100

        return outcome
