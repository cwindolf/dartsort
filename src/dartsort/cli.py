import numpy as np
import argparse
import spikeinterface.core as sc

from .util import cli_util, internal_config
from . import config, main


def dartsort_cli():
    """dartsort's CLI, invoked by `dartsort` at the terminal.

    Try `dartsort --help` to start.

    --<!!> Not stable.

    I am figuring out how to do preprocessing still. It may be configured?
    """
    # -- define CLI
    ap = argparse.ArgumentParser(
        prog="dartsort",
        epilog=dartsort_cli.__doc__.split("--")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("recording", help="Path to SpikeInterface RecordingExtractor.")
    ap.add_argument("output_directory", help="Folder where outputs will be saved.")
    ap.add_argument(
        "--config-toml",
        type=str,
        default=None,
        help="Path to configuration in TOML format. Arguments passed on the "
        "command line will override their values in the TOML file.",
    )
    # user-facing API
    cli_util.dataclass_to_argparse(config.DARTsortUserConfig, parser=ap)

    # super secret developer-only args
    dev_args = ap.add_argument_group("Secret development flags ($1.50 fee to use)")
    cli_util.dataclass_to_argparse(
        config.DeveloperConfig,
        parser=dev_args,
        prefix="_",
        skipnames=cli_util.fieldnames(config.DARTsortUserConfig),
    )

    # -- parse args
    args = ap.parse_args()

    # load the recording
    # TODO: preprocessing management
    rec = sc.load_extractor(cli_util.ensurepath(args.recording))

    # determine the config from the command line args
    cfg = cli_util.combine_toml_and_argv(
        (config.DARTsortUserConfig, config.DeveloperConfig),
        config.DeveloperConfig,
        cli_util.ensurepath(args.config_toml),
        args,
    )

    # -- run
    # TODO: maybe this should dump to Phy?
    output_directory = cli_util.ensurepath(args.output_directory, strict=False)
    ret = main.dartsort(rec, output_directory, cfg=cfg, return_extra=cfg.needs_extra)
    main.run_dev_tasks(ret, output_directory, cfg)
