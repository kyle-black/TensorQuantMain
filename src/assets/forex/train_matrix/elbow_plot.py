import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def plot_pca(df):
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                        'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI','SpreadOC','SpreadLH']
    '''
    feature_cols =['Open', 'High', 'Low', 'Close', 'USDCNH_60_Open', 'USDCNH_60_High',
       'USDCNH_60_Low', 'USDCNH_60_Close', 'USDCNH_60_Volume',
       'USDCNH_60_Volume_MA', 'NATGAS_60_Open', 'NATGAS_60_High',
       'NATGAS_60_Low', 'NATGAS_60_Close', 'NATGAS_60_Volume',
       'NATGAS_60_Volume_MA', 'COPPER_60_Open', 'COPPER_60_High',
       'COPPER_60_Low', 'COPPER_60_Close', 'COPPER_60_Volume',
       'COPPER_60_Volume_MA', 'USDJPY_60_Open', 'USDJPY_60_High',
       'USDJPY_60_Low', 'USDJPY_60_Close', 'USDJPY_60_Volume',
       'USDJPY_60_Volume_MA', 'GBPUSD_60_Open', 'GBPUSD_60_High',
       'GBPUSD_60_Low', 'GBPUSD_60_Close', 'GBPUSD_60_Volume',
       'GBPUSD_60_Volume_MA', 'CRUDEOIL_60_Open', 'CRUDEOIL_60_High',
       'CRUDEOIL_60_Low', 'CRUDEOIL_60_Close', 'CRUDEOIL_60_Volume',
       'CRUDEOIL_60_Volume_MA', 'SILVER_60_Open', 'SILVER_60_High',
       'SILVER_60_Low', 'SILVER_60_Close', 'SILVER_60_Volume',
       'SILVER_60_Volume_MA', 'EURGBP_60_Open', 'EURGBP_60_High',
       'EURGBP_60_Low', 'EURGBP_60_Close', 'EURGBP_60_Volume',
       'EURGBP_60_Volume_MA', 'USDCHF_60_Open', 'USDCHF_60_High',
       'USDCHF_60_Low', 'USDCHF_60_Close', 'USDCHF_60_Volume',
       'USDCHF_60_Volume_MA', 'GOLD_60_Open', 'GOLD_60_High', 'GOLD_60_Low',
       'GOLD_60_Close', 'GOLD_60_Volume', 'GOLD_60_Volume_MA',
       'USDCAD_60_Open', 'USDCAD_60_High', 'USDCAD_60_Low', 'USDCAD_60_Close',
       'USDCAD_60_Volume', 'USDCAD_60_Volume_MA', 'AUDUSD_60_Open',
       'AUDUSD_60_High', 'AUDUSD_60_Low', 'AUDUSD_60_Close',
       'AUDUSD_60_Volume', 'AUDUSD_60_Volume_MA', 'USDHKD_60_Open',
       'USDHKD_60_High', 'USDHKD_60_Low', 'USDHKD_60_Close',
       'USDHKD_60_Volume', 'USDHKD_60_Volume_MA', 'day_of_year', 'sine_date','Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
       'Log_Returns', 'SpreadOC', 'SpreadLH', 'MACD', 'Signal_Line_MACD',
       'RSI', 'SMI']
       '''

    X = df[feature_cols]
    X= X[50:]
    # Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Fit PCA without specifying the number of components
    pca = PCA()
    pca.fit(X_standardized)

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 5))

    # Scree plot
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')

    # Cumulative explained variance plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Plot')

    plt.tight_layout()
   # plt.show()
    plt.savefig('explained_variance_plot.png')