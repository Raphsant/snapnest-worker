"""Pre-generated asset library: catalog load, lookup, and prompt formatting.

The library replaces per-clip Higgsfield generation of hooks and outros with a
fixed pool of pre-generated S3 assets. The catalog is a single JSON object at
``{library_prefix}catalog.json`` inside the pipeline bucket; its top-level
``notes`` carry the operator-authored selection rules verbatim.

Consumers:
  - creative: selects assets per clip (``format_for_prompt`` feeds the model).
  - lint: validates that selected asset ids exist and have the right type.
  - assemble: resolves an asset id to its S3 key for download.

Pure data structures plus one thin S3 read — no DB, Anthropic, or Higgsfield
side effects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

ASSET_TYPES: tuple[str, ...] = ("hook", "outro")


@dataclass(frozen=True)
class LibraryAsset:
    """One pre-generated library asset as described by the catalog."""

    id: str
    type: str
    s3_key: str
    duration_s: float
    category: tuple[str, ...]
    tags: tuple[str, ...]
    character: str | None
    description: str
    times_used: int
    # Only meaningful for outros; hooks never carry a baked logo.
    logo_baked: bool = False


class LibraryCatalog:
    """Validated, immutable view of the library catalog."""

    def __init__(
        self,
        *,
        version: int,
        updated_at: str,
        notes: str,
        assets: dict[str, LibraryAsset],
    ) -> None:
        self.version = version
        self.updated_at = updated_at
        self.notes = notes
        self._assets = assets

    @classmethod
    def from_s3(
        cls, s3_client: S3Client, bucket: str, prefix: str
    ) -> LibraryCatalog:
        """Fetch and validate ``{prefix}catalog.json`` from the bucket."""

        key = f"{prefix}catalog.json"
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read()
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"library catalog s3://{bucket}/{key} is not valid JSON: {exc}"
            ) from exc
        catalog = cls.from_dict(parsed)
        logger.info(
            "library: loaded catalog version=%d from s3://%s/%s "
            "(%d hooks, %d outros)",
            catalog.version,
            bucket,
            key,
            len(catalog.hooks()),
            len(catalog.outros()),
        )
        return catalog

    @classmethod
    def from_dict(cls, data: object) -> LibraryCatalog:
        """Validate the raw catalog object strictly; raise ValueError on defects."""

        if not isinstance(data, dict):
            raise ValueError("library catalog must be a JSON object")
        raw = cast(dict[str, Any], data)

        version = raw.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("library catalog version must be an integer")
        updated_at = raw.get("updated_at")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("library catalog updated_at must be a non-empty string")
        notes = raw.get("notes")
        if not isinstance(notes, str):
            raise ValueError("library catalog notes must be a string")

        raw_assets = raw.get("assets")
        if not isinstance(raw_assets, list):
            raise ValueError("library catalog assets must be a list")

        assets: dict[str, LibraryAsset] = {}
        for index, raw_asset in enumerate(raw_assets):
            asset = _validate_asset(raw_asset, index)
            if asset.id in assets:
                raise ValueError(
                    f"library catalog assets[{index}] duplicates id {asset.id!r}"
                )
            assets[asset.id] = asset

        return cls(
            version=version,
            updated_at=updated_at,
            notes=notes,
            assets=assets,
        )

    def get(self, asset_id: str) -> LibraryAsset | None:
        """Return the asset with this id, or None if the catalog lacks it."""

        return self._assets.get(asset_id)

    def hooks(self) -> list[LibraryAsset]:
        """All hook assets in catalog order."""

        return [a for a in self._assets.values() if a.type == "hook"]

    def outros(self) -> list[LibraryAsset]:
        """All outro assets in catalog order."""

        return [a for a in self._assets.values() if a.type == "outro"]


def format_for_prompt(catalog: LibraryCatalog) -> str:
    """Render the catalog as compact per-asset lines plus the notes verbatim.

    One line per asset:
        H03 [hook] (mindset) tags: psychology,intensity,edu — <description>
    followed by a blank line and the catalog's selection-rule notes unchanged.
    """

    lines = [
        f"{asset.id} [{asset.type}] ({','.join(asset.category)}) "
        f"tags: {','.join(asset.tags)} — {asset.description}"
        for asset in [*catalog.hooks(), *catalog.outros()]
    ]
    return "\n".join(lines) + "\n\n" + catalog.notes


def _validate_asset(raw_asset: object, index: int) -> LibraryAsset:
    if not isinstance(raw_asset, dict):
        raise ValueError(f"library catalog assets[{index}] must be an object")
    asset = cast(dict[str, Any], raw_asset)

    asset_id = asset.get("id")
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError(
            f"library catalog assets[{index}].id must be a non-empty string"
        )

    asset_type = asset.get("type")
    if asset_type not in ASSET_TYPES:
        raise ValueError(
            f"library catalog asset {asset_id!r} has unknown type "
            f"{asset_type!r} (expected one of {ASSET_TYPES})"
        )

    s3_key = asset.get("s3_key")
    if not isinstance(s3_key, str) or not s3_key:
        raise ValueError(
            f"library catalog asset {asset_id!r} s3_key must be a "
            "non-empty string"
        )

    duration = asset.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(
            f"library catalog asset {asset_id!r} duration_s must be a number"
        )
    if duration <= 0:
        raise ValueError(
            f"library catalog asset {asset_id!r} duration_s must be positive"
        )

    category = _string_tuple(asset.get("category"), asset_id, "category")
    tags = _string_tuple(asset.get("tags"), asset_id, "tags")

    character = asset.get("character")
    if character is not None and not isinstance(character, str):
        raise ValueError(
            f"library catalog asset {asset_id!r} character must be a "
            "string or null"
        )

    description = asset.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError(
            f"library catalog asset {asset_id!r} description must be a "
            "non-empty string"
        )

    times_used = asset.get("times_used")
    if isinstance(times_used, bool) or not isinstance(times_used, int):
        raise ValueError(
            f"library catalog asset {asset_id!r} times_used must be an integer"
        )

    logo_baked = asset.get("logo_baked", False)
    if not isinstance(logo_baked, bool):
        raise ValueError(
            f"library catalog asset {asset_id!r} logo_baked must be a boolean"
        )

    return LibraryAsset(
        id=asset_id,
        type=asset_type,
        s3_key=s3_key,
        duration_s=float(duration),
        category=category,
        tags=tags,
        character=character,
        description=description,
        times_used=times_used,
        logo_baked=logo_baked,
    )


def _string_tuple(value: object, asset_id: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(
            f"library catalog asset {asset_id!r} {field} must be a list of "
            "non-empty strings"
        )
    return tuple(cast(list[str], value))
