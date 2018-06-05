import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--n-iterations-critic', type=int, default=100)
parser.add_argument('--iw', type=str, default='quadratic')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n-iterations', type=int, default=1000)
parser.add_argument('--n-perturbations', type=int, default=100)
parser.add_argument('--batch-size-c', type=int, default=50)
parser.add_argument('--batch-size-critic', type=int, default=1)
parser.add_argument('--std', type=float, default=1e-2)
parser.add_argument('--tau', type=float, default=4e-2)
parser.add_argument('--topk', type=int, default=0)
args = parser.parse_args()

keys = sorted(vars(args).keys())
run_id = 'tester-' + '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys)
print(run_id)
time.sleep(5)
