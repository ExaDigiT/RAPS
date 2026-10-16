"""
The sandbox: the one real directory the replay shell may touch.

Every path a command receives goes through ``Sandbox.resolve``; commands must never call ``open``,
``Path`` or ``os`` on a user-supplied string directly. Confinement is checked on the *fully resolved*
path (``os.path.realpath``), so ``..`` climbing out, absolute paths elsewhere and symlinks whose target is
outside the root are all rejected, as are paths that merely start with the root's name (``/sandbox-evil``).

Limits: this is not a defence against another process changing the tree between ``resolve`` and the
file operation (a check-then-use race); the sandbox directory should belong to the user running RAPS.
"""
import os
from pathlib import Path


class SandboxError(Exception):
    """A path left the sandbox (or was otherwise unusable). The message is safe to show the user."""


class Sandbox:
    def __init__(self, root, create=False):
        root = Path(root).expanduser()
        if create:
            root.mkdir(parents=True, exist_ok=True)
        self.root = Path(os.path.realpath(root))
        if not self.root.is_dir():
            raise SandboxError(f"sandbox root is not a directory: {self.root}")
        self.cwd = self.root   # always a resolved path inside the root

    def _inside(self, real):
        return real == self.root or self.root in real.parents

    def resolve(self, path, base=None):
        """The real, absolute path for ``path`` (relative to the working directory unless absolute).

        The target need not exist (for mkdir, redirection), but every existing component, including
        symlinks, is resolved first. Raises ``SandboxError`` if the result is outside the root.
        """
        path = str(path)
        if "\0" in path:
            raise SandboxError("invalid path")
        if path == "~" or path.startswith("~/"):
            path = os.path.join(str(self.root), path[2:])   # `~` is the sandbox root, the shell's home
        candidate = os.path.join(str(base or self.cwd), path)   # an absolute `path` replaces the base
        real = Path(os.path.realpath(candidate))
        if not self._inside(real):
            raise SandboxError(f"{path}: outside the sandbox")
        return real

    def display(self, real):
        """A resolved path as the user sees it: ``~`` is the sandbox root."""
        rel = Path(real).relative_to(self.root)
        return "~/" + rel.as_posix() if rel.parts else "~"

    @property
    def pwd(self):
        return self.display(self.cwd)

    def chdir(self, path):
        real = self.resolve(path)
        if not real.is_dir():
            raise SandboxError(f"{path}: not a directory")
        self.cwd = real
        return real
