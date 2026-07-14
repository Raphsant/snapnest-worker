"""Per-job working directory and S3 transfer helpers.

Each job gets an isolated directory at ``<root>/<jobId>`` (root defaults to
/tmp/jobs). Use it as a context manager so the directory is always removed when
the job finishes, success or failure.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


@dataclass
class Workspace:
    """A scratch directory for one job, plus S3 up/download helpers."""

    job_id: str
    root: Path
    s3: S3Client
    bucket: str

    @property
    def dir(self) -> Path:
        return self.root / self.job_id

    def __enter__(self) -> Workspace:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name: str) -> Path:
        """Absolute path for a file inside the job directory."""

        return self.dir / name

    def download(self, s3_key: str, name: str) -> Path:
        """Download an S3 object into the job directory; return its local path."""

        dest = self.path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, s3_key, str(dest))
        return dest

    def upload(self, local_path: Path, s3_key: str) -> None:
        """Upload a local file to S3 under the given key."""

        self.s3.upload_file(str(local_path), self.bucket, s3_key)
