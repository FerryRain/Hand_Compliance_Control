"""Read checkpoint metadata saved with path objects from a different OS."""

import os
import pickle
from pathlib import PurePosixPath, PureWindowsPath


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib":
            if os.name == "nt" and name == "PosixPath":
                return PurePosixPath
            if os.name != "nt" and name == "WindowsPath":
                return PureWindowsPath
        return super().find_class(module, name)


def load(stream, **kwargs):
    return Unpickler(stream, **kwargs).load()
