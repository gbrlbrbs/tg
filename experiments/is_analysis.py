import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=int, help='Experiment number')

def main():
    args = parser.parse_args()
    exp_num = args.exp

    original_data_path = Path('./pyhard/sjc_internacao')
    coordinates_path = original_data_path / 'coordinates.csv'
    coordinates = pd.read_csv(coordinates_path)
    generated_data_path = Path(f'./pyhard/exp{exp_num}/')
    is_exp_path = generated_data_path / 'is_with_gen_points.csv'
    all_coords_path = generated_data_path / 'all_coordinates.csv'
    figpath = Path(f'./exp{exp_num}/figs')
    figpath.mkdir(parents=True, exist_ok=True)

    enc_points = pd.read_csv(generated_data_path / 'enc_points.csv')
    calc_coords = pd.read_csv(generated_data_path / 'calc_coordinates.csv')
    points = pd.read_csv(generated_data_path / 'points.csv')

    points = points.merge(calc_coords, left_index=True, right_index=True)
    enc_points = enc_points.merge(calc_coords, left_index=True, right_index=True)
    
    points['se_z1'] = (points['z_1_x'] - points['z_1_y']) ** 2
    points['se_z2'] = (points['z_2_x'] - points['z_2_y']) ** 2

    enc_points['se_z1'] = (enc_points['z_1_x'] - enc_points['z_1_y']) ** 2
    enc_points['se_z2'] = (enc_points['z_2_x'] - enc_points['z_2_y']) ** 2

    sns.set_theme()
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Scatterplot of calculated IS points from generated points')
    sns.scatterplot(points, x='z_1_x', y='z_1_y', c=points['se_z1'], ax=axs[0], palette='viridis')
    axs[0].set_xlabel('z_1 generated')
    axs[0].set_ylabel('z_1 calculated')
    axs[0].set_title('z_1')
    plt.colorbar(axs[0].collections[0], ax=axs[0], label='Squared Error')

    sns.scatterplot(points, x='z_2_x', y='z_2_y', c=points['se_z2'], ax=axs[1], palette='viridis')
    axs[1].set_xlabel('z_2 generated')
    axs[1].set_ylabel('z_2 calculated')
    axs[1].set_title('z_2')
    plt.colorbar(axs[1].collections[0], ax=axs[1], label='Squared Error')

    fig.savefig(figpath / f'scatterplot{exp_num}.png', format='png')
    fig.savefig(figpath / f'scatterplot{exp_num}.eps', format='eps')
    plt.clf()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Scatterplot of calculated IS points from encoded points')
    sns.scatterplot(enc_points, x='z_1_x', y='z_1_y', c=enc_points['se_z1'], ax=axs[0], palette='viridis')
    axs[0].set_xlabel('z_1 encoded')
    axs[0].set_ylabel('z_1 calculated')
    axs[0].set_title('z_1')
    plt.colorbar(axs[0].collections[0], ax=axs[0], label='Squared Error')

    sns.scatterplot(enc_points, x='z_2_x', y='z_2_y', c=enc_points['se_z2'], ax=axs[1], palette='viridis')
    axs[1].set_xlabel('z_2 encoded')
    axs[1].set_ylabel('z_2 calculated')
    axs[1].set_title('z_2')
    plt.colorbar(axs[1].collections[0], ax=axs[1], label='Squared Error')

    fig.savefig(figpath / f'scatterplot_enc{exp_num}.png', format='png')
    fig.savefig(figpath / f'scatterplot_enc{exp_num}.eps', format='eps')
    plt.clf()

    is_exp = pd.read_csv(is_exp_path)
    all_coords = pd.read_csv(all_coords_path)

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    sns.scatterplot(is_exp, x='z_1', y='z_2', hue='Generated')
    plt.savefig(figpath / f'is_exp{exp_num}.png', format='png')
    plt.savefig(figpath / f'is_exp{exp_num}.eps', format='eps')
    plt.clf()

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    sns.scatterplot(all_coords, x='z_1', y='z_2', hue='Generated')
    plt.savefig(figpath / f'all_coords{exp_num}.png', format='png')
    plt.savefig(figpath / f'all_coords{exp_num}.eps', format='eps')
    plt.clf()

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    sns.scatterplot(coordinates, x='z_1', y='z_2')
    plt.savefig(figpath / f'coordinates{exp_num}.png', format='png')
    plt.savefig(figpath / f'coordinates{exp_num}.eps', format='eps')
    plt.clf()

if __name__ == '__main__':
    main()

