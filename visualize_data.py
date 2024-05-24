from matplotlib import pyplot as plt
import polars as pl
import seaborn as sns


def main():
    schedule = pl.read_csv('./data/schedule.csv')
    schedule = schedule.with_columns(
        pl.arange(end=len(schedule)).alias('index'),
    )

    sns.barplot(schedule.to_pandas(), x='day', y='cost')
    plt.show()


if __name__ == '__main__':
    main()
