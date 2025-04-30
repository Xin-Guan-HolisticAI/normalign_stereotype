import os
from pathlib import Path as PathLibPath
from typing import Union, Literal, Optional, List
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))


class Path:
    """A class for handling paths relative to different base directories."""
    
    def __init__(self, path: Union[str, PathLibPath], base: Literal["current", "project", "absolute"] = "current"):
        """
        Initialize a Path object.
        
        Args:
            path: The path string or Path object
            base: The base directory to resolve the path from:
                - "current": Resolve from current directory
                - "project": Resolve from project root
                - "absolute": Treat as absolute path
        """
        self._path = str(path)
        self._base = base
        
    def resolve(self) -> str:
        """Resolve the path to an absolute path based on the specified base."""
        if self._base == "absolute":
            return os.path.abspath(self._path)
        elif self._base == "project":
            return os.path.abspath(os.path.join(PROJECT_ROOT, self._path))
        else:  # current
            return os.path.abspath(os.path.join(CURRENT_DIR, self._path))
    
    def __str__(self) -> str:
        return self.resolve()
    
    def __repr__(self) -> str:
        return f"Path('{self._path}', base='{self._base}')"
    
    def __truediv__(self, other: Union[str, PathLibPath]) -> 'Path':
        """Support the / operator for path joining."""
        return Path(os.path.join(self._path, str(other)), base=self._base)
    
    def __eq__(self, other: object) -> bool:
        """Compare two Path objects for equality."""
        if not isinstance(other, Path):
            return False
        return self.resolve() == other.resolve()
    
    def read_text(self, encoding: str = 'utf-8') -> str:
        """Read the contents of the file as text."""
        with open(self.resolve(), 'r', encoding=encoding) as f:
            return f.read()
    
    def write_text(self, content: str, encoding: str = 'utf-8') -> None:
        """Write text content to the file."""
        with open(self.resolve(), 'w', encoding=encoding) as f:
            f.write(content)
    
    def read_json(self) -> Union[dict, list]:
        """Read and parse JSON content from the file."""
        return json.loads(self.read_text())
    
    def write_json(self, data: Union[dict, list], indent: Optional[int] = 2) -> None:
        """Write data as JSON to the file."""
        self.write_text(json.dumps(data, indent=indent))
    
    @property
    def parent(self) -> 'Path':
        """Get the parent directory of this path."""
        return Path(os.path.dirname(self._path), base=self._base)
    
    @property
    def name(self) -> str:
        """Get the name of the file or directory."""
        return os.path.basename(self._path)
    
    @property
    def stem(self) -> str:
        """Get the stem of the file (name without extension)."""
        return os.path.splitext(self.name)[0]
    
    @property
    def suffix(self) -> str:
        """Get the file extension."""
        return os.path.splitext(self.name)[1]
    
    def exists(self) -> bool:
        """Check if the path exists."""
        return os.path.exists(self.resolve())
    
    def is_file(self) -> bool:
        """Check if the path is a file."""
        return os.path.isfile(self.resolve())
    
    def is_dir(self) -> bool:
        """Check if the path is a directory."""
        return os.path.isdir(self.resolve())
    
    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory at this path."""
        path = self.resolve()
        if parents:
            os.makedirs(path, exist_ok=exist_ok)
        else:
            if exist_ok and self.exists():
                return
            os.mkdir(path)
    
    def glob(self, pattern: str) -> List['Path']:
        """Find all paths matching the given pattern."""
        import glob
        return [Path(p, base=self._base) for p in glob.glob(os.path.join(self.resolve(), pattern))]
    
    def rglob(self, pattern: str) -> List['Path']:
        """Recursively find all paths matching the given pattern."""
        import glob
        return [Path(p, base=self._base) for p in glob.glob(os.path.join(self.resolve(), '**', pattern), recursive=True)]
    
    def with_suffix(self, suffix: str) -> 'Path':
        """Return a new path with the file extension changed."""
        return Path(os.path.splitext(self._path)[0] + suffix, base=self._base)
    
    def with_name(self, name: str) -> 'Path':
        """Return a new path with the filename changed."""
        return Path(os.path.join(os.path.dirname(self._path), name), base=self._base)