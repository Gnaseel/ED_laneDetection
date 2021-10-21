import argparse
from engine.engine import EngineTheRun

def parse_arg():
    parser = argparse.ArgumentParser(description='the parser')
    parser.add_argument('--mode', help='t = train, i = inference')
    parser.add_argument('--show',action='store_true', help='Whether to show the image')
    parser.add_argument('--showAll',action='store_true', help='Whether to show the all image')
    parser.add_argument('--model_path', help='The path of pth file')
    parser.add_argument('--image_path', help='The path of image file or folder')
    parser.add_argument('--save_path', default = 'VGG16', help='The path of output image inferenced')
    parser.add_argument('--backbone', default = 'VGG16', help='The backbone network')
    parser.add_argument('--device', default = '-1', help='-1 = CPU, 0,1,2... = GPU')

    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    
    runner = EngineTheRun(args)
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'inference':
        runner.inference()
        return
    elif args.mode == 'score':
        runner.scoring()
        return

    

if __name__ == "__main__":
    # print(torch.__version__)
    # print(np.__version__)
    main()