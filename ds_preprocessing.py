import pandas as pd
import argparse
import os


def load_data(path):
    df = pd.read_csv(path, sep=',', header=None, low_memory=False, error_bad_lines=False)
    df = df.sample(frac=1).reset_index(drop=True)

    df_training = df.iloc[:int(len(df) * 0.7)]
    df_validation = df.iloc[int(len(df) * 0.7):int(len(df) * 0.9)]
    df_test = df.iloc[int(len(df) * 0.9):]

    basename = os.path.basename(path).replace('.csv', "")
    dir = os.path.dirname(path)

    df_training.to_csv(f'{dir}/{basename}_training.csv', header=False, index=False)
    df_validation.to_csv(f'{dir}/{basename}_validation.csv', header=False, index=False)
    df_test.to_csv(f'{dir}/{basename}_test.csv', header=False, index=False)

    print('Training dataset has [{}] rows and [{}] columns'.format(df_training.shape[0], df_training.shape[1]))
    print('Validation dataset has [{}] rows and [{}] columns'.format(df_validation.shape[0], df_validation.shape[1]))
    print('Test dataset has [{}] rows and [{}] columns'.format(df_test.shape[0], df_test.shape[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the path to the dataset to preprocess.')
    parser.add_argument('--filename', '-f', type=str, help='Path to the dataset to preprocess.')

    args = parser.parse_args()

    filename = args.filename

    load_data(filename)
