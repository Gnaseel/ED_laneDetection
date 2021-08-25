import argparse
from tool.trainer import Trainer

def parse_arg():
    parser = argparse.ArgumentParser(description='the parser')
    parser.add_argument('--mode', help='t = train, i = inference')
    args = parser.parse_args()
    return args
def main():
    args = parse_arg()
    print("{}  {}".format(args.mode, args.mode))
    
    if args.mode == 't':
        runner = Trainer()
        runner.train()

    

if __name__ == "__main__":
    # print(torch.__version__)
    # print(np.__version__)
    main()