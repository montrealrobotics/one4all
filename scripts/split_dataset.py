import argparse

from src.utils.split_dataset import split_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Split folder with trajectories into train and validation split.')

    parser.add_argument("--split", "-s", type=float, action='store',
                        help="Percentage of trajectories in validation set", default=0.3, required=False)
    parser.add_argument("--path", "-p", type=str, action='store',
                        help="Path to dataset", default='./data/omaze_random', required=False)

    args = parser.parse_args()
    split_data(**vars(args))
