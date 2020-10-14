from dataloader import get_dataloaders
from args import arg_parser
args = arg_parser.parse_args()

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

train_loader, val_loader, test_loader = get_dataloaders(args)
for i, (input, target) in enumerate(val_loader):
    print("i:",i)
    print("input:",input)
    print("target:",target)
