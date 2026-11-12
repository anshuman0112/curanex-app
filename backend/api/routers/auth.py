"""Sign-up, login, logout and "who am I". Doctors start as pending until the admin approves them by hand."""

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.auth import store
from api.auth.deps import current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class Signup(BaseModel):
    email: str = Field(min_length=5, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=200)
    name: str = Field(min_length=2, max_length=120)
    role: str = Field(pattern="^(patient|doctor)$")
    clinic: str | None = Field(default=None, max_length=200)
    specialty: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=30)


class Login(BaseModel):
    email: str
    password: str


@router.post("/signup")
def signup(body: Signup) -> dict:
    details = {k: v for k, v in {"clinic": body.clinic, "specialty": body.specialty, "phone": body.phone}.items() if v}
    try:
        user = store.create_user(body.email, body.password, body.name, body.role, details)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    token, user = store.login(body.email, body.password)
    return {"token": token, "user": user}


@router.post("/login")
def login(body: Login) -> dict:
    got = store.login(body.email, body.password)
    if not got:
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    return {"token": got[0], "user": got[1]}


@router.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    return user


@router.post("/logout")
def logout(authorization: str | None = Header(default=None)) -> dict:
    if authorization and authorization.lower().startswith("bearer "):
        store.logout(authorization.split(" ", 1)[1].strip())
    return {"ok": True}
