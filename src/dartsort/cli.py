import argparse
import spikeinterface.core as sc
import logging

from .util import cli_util
from . import config, main


def dartsort_cli():
    """dartsort's CLI, invoked by `dartsort` at the terminal.

    Try `dartsort --help` to start.

    ---<!!> Not stable.

    I am figuring out how to do preprocessing still. It may be configured?
    """
    # -- define CLI
    ap = argparse.ArgumentParser(
        prog="dartsort",
        epilog=dartsort_cli.__doc__.split("---")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("recording", help="Path to SpikeInterface RecordingExtractor.")
    ap.add_argument(
        "output_dir",
        nargs="?",
        help="Folder where outputs will be saved. If this is unset, then "
        "--config-toml must be supplied, and the output folder will be the one where "
        "that configuration lives. If both are supplied, output_dir will be "
        "used (not the config file's parent dir).",
    )
    ap.add_argument(
        "--config-toml",
        type=str,
        default=None,
        help="Path to configuration in TOML format. Arguments passed on the "
        "command line will override their values in the TOML file.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--loglevel",
        default="WARNING",
        type=str.upper,
        choices=logging._nameToLevel.keys(),
    )
    # user-facing API
    cli_util.dataclass_to_argparse(config.DARTsortUserConfig, parser=ap)

    # super secret developer-only args
    dev_args = ap.add_argument_group("Secret development flags ($1.50 fee to use)")
    cli_util.dataclass_to_argparse(
        config.DeveloperConfig,
        parser=dev_args,
        skipnames=cli_util.fieldnames(config.DARTsortUserConfig),
    )

    # -- parse args
    args = ap.parse_args()

    # set up logging
    logging.basicConfig(level=args.loglevel)

    # check if we have config file
    config_toml = None
    if args.config_toml:
        config_toml = cli_util.ensurepath(args.config_toml)

    # determine output directory
    if args.output_dir:
        output_dir = cli_util.ensurepath(args.output_dir, strict=False)
    elif config_toml is None:
        print(f"No output directory given, exiting. See `{ap.prog} -h`.")
        return 1
    else:
        output_dir = config_toml.parent

    # determine the config from the command line args
    cfg = cli_util.combine_toml_and_argv(
        (config.DARTsortUserConfig, config.DeveloperConfig),
        config.DeveloperConfig,
        config_toml,
        args,
    )

    # load the recording
    # TODO: preprocessing management
    try:
        rec = sc.load_extractor(cli_util.ensurepath(args.recording))
    except FileNotFoundError as e:
        ee = FileNotFoundError(
            f"The recording path passed to {ap.prog}, '{args.recording}', doesn't exist."
        )
        raise ee from e
    except Exception as e:
        ee = ValueError(
            f"{ap.prog} couldn't load your recording. Detailed error above."
        )
        raise ee from e

    # -- run
    # TODO: maybe this script should dump to Phy?
    ret = main.dartsort(
        rec,
        output_dir,
        cfg=cfg,
        overwrite=args.overwrite,
    )
