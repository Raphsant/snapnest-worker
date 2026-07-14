"""SnapNest pipeline worker.

A plain-Python (no framework) SQS consumer that runs video-processing stages
against jobs produced by the NestJS backend and writes status back to Postgres.
"""

__version__ = "0.1.0"
