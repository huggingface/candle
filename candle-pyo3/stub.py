# See: https://raw.githubusercontent.com/huggingface/tokenizers/main/bindings/python/stub.py
import argparse
import inspect
import os
from typing import Optional
import black
from pathlib import Path
import re


INDENT = " " * 4
GENERATED_COMMENT = "# Generated content DO NOT EDIT\n"
TYPING = """from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
"""
CANDLE_SPECIFIC_TYPING = "from candle.typing import _ArrayLike, Device, Scalar, Index, Shape\n"
CANDLE_TENSOR_IMPORTS = "from candle import Tensor,DType,QTensor\n"
RETURN_TYPE_MARKER = "&RETURNS&: "
ADDITIONAL_TYPEHINTS = {}
FORWARD_REF_PATTERN = re.compile(r"ForwardRef\('([^']+)'\)")


def do_indent(text: Optional[str], indent: str):
    if text is None:
        return ""
    return text.replace("\n", f"\n{indent}")


def function(obj, indent: str, text_signature: str = None):
    if text_signature is None:
        text_signature = obj.__text_signature__

    text_signature = text_signature.replace("$self", "self").lstrip().rstrip()
    doc_string = obj.__doc__
    if doc_string is None:
        doc_string = ""

    # Check if we have a return type annotation in the docstring
    return_type = None
    doc_lines = doc_string.split("\n")
    if doc_lines[-1].lstrip().startswith(RETURN_TYPE_MARKER):
        # Extract the return type and remove it from the docstring
        return_type = doc_lines[-1].lstrip()[len(RETURN_TYPE_MARKER) :].strip()
        doc_string = "\n".join(doc_lines[:-1])

    string = ""
    if return_type:
        string += f"{indent}def {obj.__name__}{text_signature} -> {return_type}:\n"
    else:
        string += f"{indent}def {obj.__name__}{text_signature}:\n"
    indent += INDENT
    string += f'{indent}"""\n'
    string += f"{indent}{do_indent(doc_string, indent)}\n"
    string += f'{indent}"""\n'
    string += f"{indent}pass\n"
    string += "\n"
    string += "\n"
    return string


def member_sort(member):
    if inspect.isclass(member):
        value = 10 + len(inspect.getmro(member))
    else:
        value = 1
    return value


def fn_predicate(obj):
    value = inspect.ismethoddescriptor(obj) or inspect.isbuiltin(obj)
    if value:
        return obj.__text_signature__ and not obj.__name__.startswith("_")
    if inspect.isgetsetdescriptor(obj):
        return not obj.__name__.startswith("_")
    return False


def get_module_members(module):
    members = [
        member
        for name, member in inspect.getmembers(module)
        if not name.startswith("_") and not inspect.ismodule(member)
    ]
    members.sort(key=member_sort)
    return members


def pyi_file(obj, indent=""):
    string = ""
    if inspect.ismodule(obj):
        string += GENERATED_COMMENT
        string += TYPING
        string += CANDLE_SPECIFIC_TYPING
        if obj.__name__ != "candle.candle":
            string += CANDLE_TENSOR_IMPORTS
        members = get_module_members(obj)
        for member in members:
            string += pyi_file(member, indent)

    elif inspect.isclass(obj):
        indent += INDENT
        mro = inspect.getmro(obj)
        if len(mro) > 2:
            inherit = f"({mro[1].__name__})"
        else:
            inherit = ""
        string += f"class {obj.__name__}{inherit}:\n"

        body = ""
        if obj.__doc__:
            body += f'{indent}"""\n{indent}{do_indent(obj.__doc__, indent)}\n{indent}"""\n'

        fns = inspect.getmembers(obj, fn_predicate)

        # Init
        if obj.__text_signature__:
            body += f"{indent}def __init__{obj.__text_signature__}:\n"
            body += f"{indent+INDENT}pass\n"
            body += "\n"

        if obj.__name__ in ADDITIONAL_TYPEHINTS:
            additional_members = inspect.getmembers(ADDITIONAL_TYPEHINTS[obj.__name__])
            additional_functions = []
            for name, member in additional_members:
                if inspect.isfunction(member):
                    additional_functions.append((name, member))

            def process_additional_function(fn):
                signature = inspect.signature(fn)
                cleaned_signature = re.sub(FORWARD_REF_PATTERN, r"\1", str(signature))
                string = f"{indent}def {fn.__name__}{cleaned_signature}:\n"
                string += (
                    f'{indent+INDENT}"""{indent+INDENT}{do_indent(fn.__doc__, indent+INDENT)}{indent+INDENT}"""\n'
                )
                string += f"{indent+INDENT}pass\n"
                string += "\n"
                return string

            for name, fn in additional_functions:
                body += process_additional_function(fn)

        for name, fn in fns:
            body += pyi_file(fn, indent=indent)

        if not body:
            body += f"{indent}pass\n"

        string += body
        string += "\n\n"

    elif inspect.isbuiltin(obj):
        string += f"{indent}@staticmethod\n"
        string += function(obj, indent)

    elif inspect.ismethoddescriptor(obj):
        string += function(obj, indent)

    elif inspect.isgetsetdescriptor(obj):
        # TODO it would be interesting to add the setter maybe ?
        string += f"{indent}@property\n"
        string += function(obj, indent, text_signature="(self)")

    elif obj.__class__.__name__ == "DType":
        string += f"class {str(obj).lower()}(DType):\n"
        string += f"{indent+INDENT}pass\n"
    else:
        raise Exception(f"Object {obj} is not supported")
    return string


def py_file(module, origin):
    members = get_module_members(module)

    string = GENERATED_COMMENT
    string += f"from .. import {origin}\n"
    string += "\n"
    for member in members:
        if hasattr(member, "__name__"):
            name = member.__name__
        else:
            name = str(member)
        string += f"{name} = {origin}.{name}\n"
    return string


def do_black(content, is_pyi):
    mode = black.Mode(
        target_versions={black.TargetVersion.PY35},
        line_length=119,
        is_pyi=is_pyi,
        string_normalization=True,
    )
    try:
        return black.format_file_contents(content, fast=True, mode=mode)
    except black.NothingChanged:
        return content


def write(module, directory, origin, check=False):
    submodules = [(name, member) for name, member in inspect.getmembers(module) if inspect.ismodule(member)]

    filename = os.path.join(directory, "__init__.pyi")
    pyi_content = pyi_file(module)
    pyi_content = do_black(pyi_content, is_pyi=True)
    os.makedirs(directory, exist_ok=True)
    if check:
        with open(filename, "r") as f:
            data = f.read()
            print("generated content")
            print(pyi_content)
            assert data == pyi_content, f"The content of {filename} seems outdated, please run `python stub.py`"
    else:
        with open(filename, "w") as f:
            f.write(pyi_content)

    filename = os.path.join(directory, "__init__.py")
    py_content = py_file(module, origin)
    py_content = do_black(py_content, is_pyi=False)
    os.makedirs(directory, exist_ok=True)

    is_auto = False
    if not os.path.exists(filename):
        is_auto = True
    else:
        with open(filename, "r") as f:
            line = f.readline()
            if line == GENERATED_COMMENT:
                is_auto = True

    if is_auto:
        if check:
            with open(filename, "r") as f:
                data = f.read()
                print("generated content")
                print(py_content)
                assert data == py_content, f"The content of {filename} seems outdated, please run `python stub.py`"
        else:
            with open(filename, "w") as f:
                f.write(py_content)

    for name, submodule in submodules:
        write(submodule, os.path.join(directory, name), f"{name}", check=check)


def extract_additional_types(module):
    additional_types = {}
    for name, member in inspect.getmembers(module):
        if inspect.isclass(member):
            if hasattr(member, "__name__"):
                name = member.__name__
            else:
                name = str(member)
            if name not in additional_types:
                additional_types[name] = member
    return additional_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    # Enable execution from the candle and candle-pyo3 directories
    cwd = Path.cwd()
    directory = "py_src/candle/"
    if cwd.name != "candle-pyo3":
        directory = f"candle-pyo3/{directory}"

    import candle
    import _additional_typing

    ADDITIONAL_TYPEHINTS = extract_additional_types(_additional_typing)

    write(candle.candle, directory, "candle", check=args.check)
