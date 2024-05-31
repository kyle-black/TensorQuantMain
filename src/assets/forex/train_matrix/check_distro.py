import matplotlib.pyplot as plt
import scipy.stats as stats


def create_plot(df, column):
   
    plt.hist(df[column], bins=100, alpha=0.5, label=f'{column}')
    plt.title('Histogram of log pct change')

    plt.xlabel('log pct change')

    plt.ylabel('Frequency')

    plt.savefig('log_histo.png')


    ## QQ plot

    stats.probplot(df['log_pct_change'], dist="norm", plot=plt)


    plt.title('QQ plot of log pct change')

    plt.savefig('qq_plot_log.png')