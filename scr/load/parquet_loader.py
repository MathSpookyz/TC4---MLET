import boto3
import io
import pandas as pd

s3 = boto3.client("s3")

def save_raw_parquet(df: pd.DataFrame, s3_path: str):
    bucket, key = _parse_s3_path(s3_path)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def save_processed_parquet(df: pd.DataFrame, s3_prefix: str):
    bucket, prefix = _parse_s3_path(s3_prefix)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}prices_processed.parquet",
        Body=buffer.getvalue()
    )


def _parse_s3_path(s3_path: str):
    s3_path = s3_path.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)
    return bucket, key