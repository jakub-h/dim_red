from plotnine import *
import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = pd.DataFrame({
        'month': range(1, 13),
        'costs': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 85000, 99000, 190000, 1100000, 1290000],
        'sales': [5064, 5503, 4890, 4800, 4154, 3932, 3699, 4245, 2890, 1567, 1309, 1254]
    })
    theme_set(theme_xkcd())
    sales_true = (ggplot(aes('month', 'sales', group=1), df) +
                  geom_line() +
                  scale_x_continuous(
                      breaks=range(1, 13),
                      limits=(1, 12),
                      labels=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) +
                  scale_y_continuous(
                      limits=(1, 6000)) +
                  ggtitle("True sales (linear axis)"))
    print(sales_true)
    costs_true = (ggplot(aes('month', 'costs', group=1), df) +
                  geom_line() +
                  scale_x_continuous(
                      breaks=range(1, 13),
                      limits=(1, 12),
                      labels=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) +
                  scale_y_continuous() +
                  ggtitle("True costs (linear axis)"))
    print(costs_true)
    sales_fake = (ggplot(aes('month', 'sales', group=1), df) +
                  geom_line() +
                  scale_x_continuous(
                      breaks=range(1, 13),
                      limits=(1, 12),
                      labels=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) +
                  scale_y_log10(
                      limits=(1, 10000)) +
                  ggtitle("Faked sales (log10 axis)"))
    print(sales_fake)
    costs_fake = (ggplot(aes('month', 'costs', group=1), df) +
                  geom_line() +
                  scale_x_continuous(
                      breaks=range(1, 13),
                      limits=(1, 12),
                      labels=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) +
                  scale_y_log10(
                      limits=(100, 1000000000)) +
                  ggtitle("Faked costs (log10 axis)"))
    print(costs_fake)
