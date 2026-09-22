"""Minimal Athena helper built on boto3 only (no extra installs on SageMaker)."""
from __future__ import annotations

import time

import pandas as pd


class Athena:
    def __init__(self, boto_session, output_location: str, database: str | None = None,
                 workgroup: str = "primary"):
        self.client = boto_session.client("athena")
        self.s3 = boto_session.client("s3")
        self.output_location = output_location
        self.database = database
        self.workgroup = workgroup

    def execute(self, sql: str, database: str | None = None, poll: float = 2.0) -> str:
        """Run a statement and block until it finishes. Returns the execution id."""
        kwargs = dict(QueryString=sql, WorkGroup=self.workgroup,
                      ResultConfiguration={"OutputLocation": self.output_location})
        db = database if database is not None else self.database
        if db:
            kwargs["QueryExecutionContext"] = {"Database": db}
        qid = self.client.start_query_execution(**kwargs)["QueryExecutionId"]
        while True:
            status = self.client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
            state = status["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(poll)
        if state != "SUCCEEDED":
            raise RuntimeError(f"Athena query {state}: {status.get('StateChangeReason')}\n{sql}")
        return qid

    def query(self, sql: str, database: str | None = None) -> pd.DataFrame:
        """Run a SELECT and return the result CSV as a DataFrame."""
        qid = self.execute(sql, database)
        path = self.client.get_query_execution(QueryExecutionId=qid)[
            "QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        bucket, key = path.replace("s3://", "").split("/", 1)
        body = self.s3.get_object(Bucket=bucket, Key=key)["Body"]
        return pd.read_csv(body)

    def scanned_mb(self, qid: str) -> float:
        stats = self.client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Statistics"]
        return stats.get("DataScannedInBytes", 0) / 1e6
