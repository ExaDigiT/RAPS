import argparse
from .download import download
from .loader import load_data

def main():
    p = argparse.ArgumentParser(prog="mit_supercloud")
    subs = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start",     default="21052021")
    common.add_argument("--end",       default="22052021")
    common.add_argument("--partition", choices=["all","part-cpu","part-gpu"], default="all")
    common.add_argument("--outdir",    default="source_data")
    common.add_argument("--bucket",    default="mit-supercloud-dataset")
    common.add_argument("--prefix",    default="datacenter-challenge/202201/")
    common.add_argument("--max-jobs",  type=int)
    common.add_argument("--dry-run",   action="store_true")

    pd = subs.add_parser("download", parents=[common], help="Fetch data from S3")
    pd.set_defaults(func=download)

    pl = subs.add_parser("load", parents=[common], help="Load local data into RAPS")
    pl.add_argument("path", help="Local data root")
    pl.set_defaults(func=lambda args: load_data(args.path,
                                                 start_date=args.start,
                                                 end_date=args.end,
                                                 partition=args.partition))

    args = p.parse_args()
    return args.func(args)

if __name__ == "__main__":
    main()
