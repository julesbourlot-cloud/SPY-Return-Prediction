import yfinance as yh
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def plot_scatter_matrix(train):
        
        scatter_matrix(train, figsize=(10,10))
        
        plt.suptitle("Scatter Matrix")
        plt.show()


def correl_matrix(train):
        full_correl = train.corr()
        plt.figure(figsize=(10,10))
        plt.imshow(full_correl)
        plt.colorbar()
        plt.xticks(
            range(len(full_correl.columns)),
            full_correl.columns,
            rotation = 45,
            ha="right"
        )
        plt.yticks(
            range(len(full_correl.columns)),
            full_correl.columns
        )
        plt.title("Correlation Matrix")
        for i in range(len(full_correl.columns)):
            for j in range(len(full_correl.columns)):
                plt.text(
                    j,
                    i,
                    round(full_correl.iloc[i,j],2),
                    ha="center",
                    va="center",
                    color="white"
                )
        
        plt.tight_layout()
        


        plt.show()

def predicted_vs_actual_spy(test, pred_japan):
        plt.figure(figsize=(15,8))
        plt.plot(test.index, test['SPY'], label="Actual")
        plt.plot(test.index, pred_japan, label="Predicted")
        plt.title("Actual versus Predicted SPY Log returns (Test Set)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.tight_layout()
        plt.show()




def main():
    tickers =[
        "^N225","^HSI","000001.SS","^KS11","SPY"
        ]

    data = yh.download(
        tickers, start="2015-01-01", end="2026-01-01", auto_adjust=True
        )

    prices = data['Close']

        
    log_returns = np.log(prices/prices.shift(1))
    log_returns = log_returns.dropna()
    log_returns = log_returns.rename(columns={
        "^N225":"N225",
        "^HSI":"HSI",
        "000001.SS":"Shanghai",
        "^KS11":"KS11"
    })

        

    train = log_returns.iloc[-2000:-1000,:].copy()
    test = log_returns.iloc[-1000:,:].copy()





    
    formula_full = 'SPY~N225+HSI+Shanghai+KS11'
    formula_japan = 'SPY~N225'                           # "best" model
    formula_japan_hk = 'SPY~N225+HSI'

    model_full = smf.ols(formula_full, data=train).fit()
    model_japan = smf.ols(formula_japan, data=train).fit()
    model_japan_hk = smf.ols(formula_japan_hk, data=train).fit()

    pred_full = model_full.predict(test)
    pred_japan = model_japan.predict(test)
    pred_japan_hk = model_japan_hk.predict(test)

    rmsefull = ((((pred_full-test['SPY'])**2).sum())/(test.shape[0]))**0.5
    rmsejapan = ((((pred_japan-test['SPY'])**2).sum())/(test.shape[0]))**0.5
    rmsejapan_hk = ((((pred_japan_hk-test['SPY'])**2).sum())/(test.shape[0]))**0.5

    SST= ((test['SPY']-test['SPY'].mean())**2).sum()   
    
    SSE1 = ((test['SPY']-pred_full)**2).sum()
    r_squared_full = 1 - (SSE1/SST) #this is japan, china, korea, and hong kong

    SSE2 = ((test['SPY']-pred_japan)**2).sum()
    r_squared_japan = 1 - (SSE2/SST) #this is just japan

    SSE3 = ((test['SPY']-pred_japan_hk)**2).sum()
    r_squared_japan_hk = 1 - (SSE3/SST) #this is japan and hong kong


        

    output = pd.DataFrame({
        "Model":[
            "Japan",
            "Japan + Hong Kong",
            "Japan + Hong Kong + Korea + China"
        ],
        "Predictors":[
            "N225",
            "N225 + HSI",
            "N225 + HSI + KS11 + Shanghai"
        ],
        "Test RMSE":[
            rmsejapan,
            rmsejapan_hk,
            rmsefull
        ],
        "Test R-Squared":[
            r_squared_japan,
            r_squared_japan_hk,
            r_squared_full
        ],
        "Train R-Squared":[
            model_japan.rsquared,
            model_japan_hk.rsquared,
            model_full.rsquared
        ],
        "Adjusted R-Squared":[
            model_japan.rsquared_adj,
            model_japan_hk.rsquared_adj,
            model_full.rsquared_adj
        ]
    })


    print(output)
    plot_scatter_matrix(train)
    correl_matrix(train)
    predicted_vs_actual_spy(test, pred_japan)

if __name__ == "__main__":
    main()
