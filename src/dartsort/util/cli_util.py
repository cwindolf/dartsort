# TODO: arg_group to group arguments in the -h.
import typing
from argparse import ArgumentParser, BooleanOptionalAction, _StoreAction
from dataclasses import _MISSING_TYPE, MISSING, asdict, field, fields
from pathlib import Path
from typing import Any, Callable

from annotated_types import Ge, Gt, Le, Lt


def ensurepath(path: str | Path, strict=True):
    path = Path(path)
    path = path.expanduser()
    path = path.resolve(strict=strict)
    return path


def argfield(
    default: Any = MISSING,
    default_factory: Callable | _MISSING_TYPE = MISSING,
    arg_type: Callable | _MISSING_TYPE = MISSING,
    cli=True,
    doc="",
):
    """Helper for defining fields with extended CLI behavior.

    This is only needed when a field's type is not a callable which can
    take string inputs and return an object of the right type, such as
    typing.Union or something. Then arg_type is what the CLI will call
    to convert the argv element into an object of the desired type.

    Fields with cli=False will not be available from the command line.
    """
    metadata: dict[str, Any] = dict(cli=cli, doc=doc)
    if arg_type is not MISSING:
        metadata["arg_type"] = arg_type
    return field(default=default, default_factory=default_factory, metadata=metadata)


def fieldnames(cls):
    return set(f.name for f in fields(cls))


def manglefieldset(name):
    return f"{name}$$fieldset"


# these two classes register that args are passed on the command line,
# giving a way to use cli args to override config file args based on
# whether they were actually passed (vs being non-default, say)


class FieldStoreAction(_StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string=option_string)
        setattr(namespace, manglefieldset(self.dest), values)


class FieldBooleanOptionalAction(BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string=option_string)
        setattr(namespace, manglefieldset(self.dest), True)


def field_annot_str(field):
    constrs = []
    for tp in field.metadata:
        if isinstance(tp, Gt):
            constrs.append(f"> {tp.gt}")
        elif isinstance(tp, Ge):
            constrs.append(f">= {tp.ge}")
        elif isinstance(tp, Lt):
            constrs.append(f"< {tp.lt}")
        elif isinstance(tp, Le):
            constrs.append(f"<= {tp.le}")
        else:
            raise ValueError(f"Haven't implemented {tp} annotation.")
    return ", ".join(constrs)


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
        doc = field.metadata.get("doc", "")
        type_ = field.metadata.get("arg_type", field.type)
        if type_ is MISSING:
            raise ValueError(f"Need type or arg_type for {field}.")

        choices = None
        if typing.get_origin(type_) == typing.Literal:
            choices = typing.get_args(type_)
            type_ = type(choices[0])
        elif typing.get_origin(type_) == typing.Annotated:
            type_, annot = typing.get_args(type_)
            annot = field_annot_str(annot)
            if annot:
                doc += f" (%(type)s; {annot})"
            else:
                doc += " (%(type)s)"
        elif type_ != bool:
            doc += " (%(type)s)"

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
                parser.add_argument(name, action=FieldBooleanOptionalAction, **kw)  # type: ignore
            else:
                parser.add_argument(
                    name,
                    action=FieldStoreAction,
                    type=type_,
                    required=required,
                    **kw,  # type: ignore
                )
        except Exception as e:
            ee = ValueError(f"Exception raised while adding {field=} to CLI")
            raise ee from e

    return parser


def dataclass_from_toml(clss, toml_path):
    import tomllib  # TODO: can hold off on py3.11 for now

    with open(toml_path, "rb") as toml:
        for j, cls in enumerate(clss):
            try:
                return cls(**tomllib.load(toml))
            except TypeError:
                if j < len(clss) - 1:
                    continue
                raise


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
