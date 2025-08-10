# root/main.py
try:
    # if your app lives in api/main.py
    from api.main import app            # noqa: F401
except ModuleNotFoundError:
    # if your app lives in app/main.py
    from app.main import app            # noqa: F401
