import argparse

def get_parameters(arg=None):
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument("--debug", action='store_true', default=False,
                        help="whether to use debug mode")
    parser.add_argument("--config_path", default='example_configs/self-attention_bs_church.',
                        help="path to the configuration file")
    if arg:
        args, _ = parser.parse_known_args(arg.split())
    else:
        args, _ = parser.parse_known_args()

    return args
