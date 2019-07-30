import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--is-largest-wf", action='store_true')

args = parser.parse_args()

def main():
   print(args.is_largest_wf)

if __name__ == '__main__':
    main()
