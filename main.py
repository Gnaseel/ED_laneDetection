import argparse
from engine.engine import EngineTheRun

def parse_arg():
    parser = argparse.ArgumentParser(description='the parser')
    parser.add_argument('--mode', help='t = train, i = inference')
    parser.add_argument('--show', help='Whether to show the image')
    args = parser.parse_args()
    return args
def main():
    args = parse_arg()
    
    runner = EngineTheRun(args)
    if args.mode == 't':
        runner.train()
    elif args.mode == 'i':
        runner.inference()
        return


    

if __name__ == "__main__":
    # print(torch.__version__)
    # print(np.__version__)
    main()