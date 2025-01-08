from pathlib import Path
from dataclasses import MISSING, fields, field, asdict
from argparse import ArgumentParser, BooleanOptionalAction, _StoreAction
import tomllib
import typing

from torch import Value


def ensurepath(path, strict=True):
    path = Path(path)
    path = path.expanduser()
    path = path.resolve(strict=strict)
    return path


def argfield(
    default=MISSING, default_factory=MISSING, arg_type=MISSING, cli=True, doc=""
):
    """Helper for defining fields with extended CLI behavior.

    This is only needed when a field's type is not a callable which can
    take string inputs and return an object of the right type, such as
    typing.Union or something. Then arg_type is what the CLI will call
    to convert the argv element into an object of the desired type.

    Fields with cli=False will not be available from the command line.
    """
    metadata = dict(cli=cli, doc=doc)
    if arg_type is not MISSING:
        metadata["arg_type"] = arg_type
    return field(default=default, default_factory=default_factory, metadata=metadata)


def fieldnames(cls):
    return set(f.name for f in fields(cls))


def manglefieldset(name):
    return f"{name}$$fieldset"


class FieldStoreAction(_StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string=option_string)
        setattr(namespace, f"{self.dest}$$fieldset", values)


class FieldBooleanOptionalAction(BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string=option_string)
        setattr(namespace, manglefieldset(self.dest), True)


def dataclass_to_argparse(cls, parser=None, prefix="", skipnames=None):
    """Add a dataclass's fields as arguments to an ArgumentParser

    Inspired by Jeremy Stafford's datacli. Works together with argfield
    to set metadata needed sometimes.
    """
    if parser is None:
        parser = ArgumentParser()

    for field in fields(cls):
        if skipnames and field.name in skipnames:
            continue
        if not field.metadata.get("cli", True):
            continue

        required = field.default is MISSING and field.default_factory is MISSING
        doc = field.metadata.get("doc", None)
        type_ = field.metadata.get("arg_type", field.type)
        if type_ is MISSING:
            raise ValueError(f"Need type or arg_type for {field}.")
        choices = None
        if typing.get_origin(type_) == typing.Literal:
            choices = typing.get_args(type_)
            type_ = type(choices[0])

        name = f"--{prefix}{field.name.replace('_', '-')}"
        metavar = field.name.upper()
        default = field.default
        if default is MISSING:
            default = None
        kw = dict(
            default=default, help=doc, metavar=metavar, dest=field.name, choices=choices
        )

        try:
            if type_ == bool:
                parser.add_argument(name, action=FieldBooleanOptionalAction, **kw)
            else:
                parser.add_argument(
                    name, action=FieldStoreAction, type=type_, required=required, **kw
                )
        except Exception as e:
            ee = ValueError(f"Exception raised while adding {field=} to CLI")
            raise ee from e

    return parser


def dataclass_from_toml(clss, toml_path):
    with open(toml_path, "r") as toml:
        for cls in clss:
            try:
                return cls(**tomllib.load(toml))
            except TypeError:
                continue


def update_dataclass_from_args(cls, obj, args):
    if obj is None:
        kv = {}
    else:
        kv = asdict(obj)

    for field in fields(cls):
        if hasattr(args, manglefieldset(field.name)):
            kv[field.name] = getattr(args, field.name)

    return cls(**kv)


def combine_toml_and_argv(toml_dataclasses, target_dataclass, toml_path, args):
    # validate the toml file, if supplied
    cfg = None
    if toml_path:
        cfg = dataclass_from_toml(toml_dataclasses, toml_path)

    # update with additional arguments
    cfg = update_dataclass_from_args(target_dataclass, cfg, args)

    return cfg
