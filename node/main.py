import argparse
from train import train

def add_fit_args(parser):

    parser.add_argument('filename')
    parser.add_argument('--dataset',type=str,default='cora',help='dataset for node classification task')
    parser.add_argument('-v','--verbose',action='store_true',help='verbosity to print out epochs and loss')
    parser.add_argument('--repetitions',type=int,default=50,help='number of repetitions of training')
    parser.add_argument('--batch_size',type=int,default=4,help='batch size for training')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate for training')
    parser.add_argument('--max_epochs',type=int,default=3000,help='max number of epochs')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    args = add_fit_args(argparse.ArgumentParser(description="Train"))

