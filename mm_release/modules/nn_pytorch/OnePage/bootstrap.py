import os, sys
sys.path.append(os.getcwd())
sys.path.append("..")

import argparse
import torch
import yaml

import importlib

from OnePage.engine_utils import dict2obj, find_class

def parse_cmd():
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument("--option", type=str, default='./options.yaml',
                                  help="option file")
    parser.add_argument("--fork", type=str, default=None,
                                  help="fork to other train branch")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")

    train_arg_parser.add_argument("--checkpoint_g", type=str, default=None,
                                  help="checkpoint G model file")

    train_arg_parser.add_argument("--checkpoint_d", type=str, default=None,
                                  help="checkpoint D model file")

    valid_arg_parser = subparsers.add_parser("valid", help="parser for validating arguments")
    valid_arg_parser.add_argument("--input", type=str, default=None,
                                  help="valid dir")
    valid_arg_parser.add_argument("--output", type=str, default=None,
                                  help="valid result dir")

    subparsers.add_parser("preprocess", help="parser for validating arguments")

    opt = parser.parse_args()
    return opt


def load_config(conf_file):
    with open(conf_file, 'r') as stream:
        try:
            opt_data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print("options config format error", exc)
            sys.exit(1)

    options = dict2obj(opt_data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options.device = device

    return options



def init_engine(args):
    options = load_config(args.option)
    options.fork = args.fork if args.fork is not None else options.train.fork

    engine_class = find_class(None, options.train.engine)
    if engine_class is None:
        engine_class = \
            getattr(importlib.import_module(".", options.train.engine),
                    options.train.engine)

        if engine_class is None:
            print("engine not found", options.train.engine)
            sys.exit(1)
    print('found engine class', engine_class)
    engine = engine_class(options)
    _set_global_engine(engine)
    importlib.import_module('OnePage.common_callbacks')
    importlib.import_module(".", options.train.callbacks)
    # check
    engine.dump_callbacks()
    print(f"Register callbacks from {options.train.callbacks} Done")

    engine.subcmd = args.subcommand
    return engine


def main():
    args = parse_cmd()

    engine = init_engine(args)

    if args.subcommand == 'train':
        engine.run()
    elif args.subcommand == 'valid':
        if args.input is not None:
            engine.opt.train.input_dir = args.input
        if args.output is not None:
            engine.opt.train.output_dir = args.output
        engine.valid()
    elif args.subcommand == 'preprocess':
        engine.preprocess()

engine = None
def _set_global_engine(e):
    global engine
    engine = e

if __name__ == "__main__":
    main()