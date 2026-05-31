from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class OKResponse(BaseModel, Generic[T]):
    ok: bool = True
    data: T


class ErrorResponse(BaseModel):
    ok: bool = False
    detail: str
    code: str | None = None


class JobReference(BaseModel):
    job_id: str
    status_url: str
