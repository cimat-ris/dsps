import numpy as np
import argparse

training_data_path = './training/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Name of subject folder.')
    parser.add_argument('--npoints', default='20', help='Number of points in each streamline. (default: 20)')
    args = parser.parse_args()

    with open(training_data_path+args.npoints+'/'+args.subject+'_0.npy', 'rb') as f:
        x_training = np.load(f)
        y_training = np.load(f)
        print(x_training.shape)
        print(y_training.shape)