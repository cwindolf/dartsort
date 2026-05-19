# TODO: arg_group to group arguments in the -h.
import types
import typing
from argparse import ArgumentParser, BooleanOptionalAction, _StoreAction
from collections.abc import Sequence
from dataclasses import _MISSING_TYPE, MISSING, asdict, field, fields
from pathlib import Path
from typing import Any, Callable, Literal

from annotated_types import Ge, Gt, Le, Lt
from typing_extensions import Doc


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


def union_arg_type(tp):
    """Returns a from_string fn for types like `int | None` or `Literal["hi", "bye"] | None`."""
    types_ = typing.get_args(tp)

    # unpack any annotated internal types
    tps = []
    for tp in types_:
        if typing.get_origin(tp) == typing.Annotated:
            tp, _ = typing.get_args(tp)
        tps.append(tp)

    # let's just exclude some bad ideas here, because I don't
    # really want to figure out how to disambiguate these cases
    assert bool not in tps
    if str in tps:
        assert len(tps) == 2
    if int in tps:
        assert float not in tps
    assert not any(typing.get_origin(tp) == typing.Union for tp in tps)
    assert not any(typing.get_origin(tp) == types.UnionType for tp in tps)

    def tp_from_str(s: str) -> tp:
        s = s.strip()

        # check for None then Literal first, then grab bag
        for pos in tps:
            if pos is None and s.lower() in ("none", ""):
                return None
        for pos in tps:
            if typing.get_origin(pos) == typing.Literal:
                for opt in typing.get_args(pos):
                    if s == opt:
                        return opt
        for pos in tps:
            if pos is int and not s.strip("0123456789"):
                return int(s)
            if pos is float:
                return float(s)
            if pos is str:
                return s

        raise ValueError(
            f"Don't know how to handle type {tp} (input value was: {s}; full type list was {tps})"
        )

    return tp_from_str


def sequence_arg_type(seq):
    (internal_tp,) = typing.get_args(seq)
    is_str = internal_tp is str
    is_str_literal = (
        typing.get_origin(internal_tp) == Literal
        and type(typing.get_args(internal_tp)[0]) is str
    )
    assert is_str or is_str_literal

    def from_str(s):
        return tuple(ss.strip() for ss in s.split(","))

    return from_str


def dataclass_to_argparse(cls, parser=None, prefix="", skipnames=None):
    """Add a dataclass's fields as arguments to an ArgumentParser

    Inspired by Jeremy Stafford's datacli. Works together with argfield
    to set metadata needed sometimes.
    """
    if parser is None:
        parser = ArgumentParser()

    for fld in fields(cls):
        if skipnames and fld.name in skipnames:
            continue
        if not fld.metadata.get("cli", True):
            continue

        # handle Annotated fields
        required = fld.default is MISSING and fld.default_factory is MISSING
        doc = fld.metadata.get("doc", "")
        type_ = fld.metadata.get("arg_type", fld.type)
        if type_ is MISSING:
            raise ValueError(f"Need type or arg_type for {fld}.")
        if typing.get_origin(type_) == typing.Annotated:
            type_, *annots = typing.get_args(type_)
            for annot in annots:
                if isinstance(annot, Doc):
                    assert not doc
                    doc = annot.documentation
            annots = [ann for ann in annots if not isinstance(ann, Doc)]
            if annots:
                assert len(annots) == 1
                typeannot = field_annot_str(annots[0])
                if typeannot:
                    doc += f" (%(type)s; {typeannot})"
                else:
                    doc += " (%(type)s)"
        elif type_ is not bool:
            doc += " (%(type)s)"

        # handle the type itself
        choices = None
        if typing.get_origin(type_) == typing.Literal:
            choices = typing.get_args(type_)
            type_ = type(choices[0])
        elif typing.get_origin(type_) in (types.UnionType, typing.Union):
            type_ = union_arg_type(type_)
        elif typing.get_origin(type_) == Sequence:
            type_ = sequence_arg_type(type_)

        name = f"--{prefix}{fld.name.replace('_', '-')}"
        metavar = fld.name.upper()
        default = fld.default
        if default is MISSING:
            default = None

        try:
            if type_ is bool:
                parser.add_argument(
                    name,
                    action=FieldBooleanOptionalAction,
                    default=default,
                    help=doc,
                    dest=fld.name,
                )
            else:
                parser.add_argument(
                    name,
                    action=FieldStoreAction,
                    type=type_,
                    required=required,
                    choices=choices,
                    metavar=metavar,
                    default=default,
                    help=doc,
                    dest=fld.name,
                )
        except Exception as e:
            ee = ValueError(f"Exception raised while adding field '{fld}' to CLI")
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

    for fld in fields(cls):
        if hasattr(args, manglefieldset(fld.name)):
            kv[fld.name] = getattr(args, fld.name)

    return cls(**kv)


def combine_toml_and_argv(toml_dataclasses, target_dataclass, toml_path, args):
    # validate the toml file, if supplied
    cfg = None
    if toml_path:
        cfg = dataclass_from_toml(toml_dataclasses, toml_path)

    # update with additional arguments
    cfg = update_dataclass_from_args(target_dataclass, cfg, args)

    return cfg
