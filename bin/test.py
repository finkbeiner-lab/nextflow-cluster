#!/opt/conda/bin/python

import argparse
print('test.py imported successfully')
def test(txt):
    print(txt)
    return txt

if __name__=='__main__':
    print('world')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--txt',
        type=str,
        help='text'
    )
    args = parser.parse_args()
    test(args.txt)
