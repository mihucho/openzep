import hmac

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings
from graphiti_core import Graphiti

bearerScheme = HTTPBearer(auto_error=False)


def get_graphiti(request: Request) -> Graphiti:
    return request.app.state.graphiti


def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearerScheme),
):
    if not settings.api_key:
        return
    # 支持 Bearer 和 Api-Key 两种格式
    token = None
    if credentials is not None:
        token = credentials.credentials
    else:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("api-key "):
            token = auth_header[8:]
    # Constant-time comparison to avoid leaking key length / prefix via timing.
    if token is None or not hmac.compare_digest(token, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or missing API key. The ZEP_API_KEY on the client side must "
                "exactly match the API_KEY in openzep/.env. If you are connecting from a "
                "Docker container, confirm ZEP_BASE_URL points at the host "
                "(host.docker.internal or the host LAN IP) and ends with /api/v2."
            ),
        )
