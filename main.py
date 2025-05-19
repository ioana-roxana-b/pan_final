from src import test
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", "--input-directory", required=True, help="Input directory")
    parser.add_argument("-o", "--output", "--output-directory", required=True, help="Output directory")
    args = parser.parse_args()

    test.test(args=args)