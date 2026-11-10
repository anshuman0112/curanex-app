from fastapi import APIRouter, HTTPException, Query

from api import alerts

router = APIRouter()


@router.get("/alerts")
def area_alert(district: str = Query("", max_length=60)):
    """Dengue-season weather for a district. Only the district name is used; nothing about the user is stored."""
    try:
        return alerts.alert(district)
    except alerts.AlertError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
