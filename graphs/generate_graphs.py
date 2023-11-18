import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def main():
    sns.set_theme()
    csv_files = Path('./').glob('*.csv')
    for file in csv_files:
        exp_num = file.name.split('_')[0][-1]
        df = pd.read_csv(file)
        ax = sns.lineplot(data=df, x='Step', y='Value')
        ax.set_title(f"Experiment {exp_num}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(f'./loss_exp{exp_num}.png', format='png')
        plt.savefig(f'./loss_exp{exp_num}.eps', format='eps')
        plt.clf()
    
if __name__ == '__main__':
    main()