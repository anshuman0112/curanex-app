"""FastAPI dependencies: who is calling, and are they allowed. Roles are checked on the server for every request."""

from fastapi import Depends, Header, HTTPException

from api.auth import store


def current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Please log in.")
    user = store.user_for_token(authorization.split(" ", 1)[1].strip())
    if not user:
        raise HTTPException(status_code=401, detail="Your session has expired. Please log in again.")
    return user


def require_role(*roles: str):
    def check(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="You don't have access to this.")
        if user["role"] == "doctor" and user["status"] != "approved":
            raise HTTPException(status_code=403, detail="Your doctor account is waiting for approval.")
        return user
    return check
