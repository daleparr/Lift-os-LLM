"""
Security utilities for Lift-os-LLM microservice.

Handles JWT authentication, password hashing, and security validation.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import settings
from .logging import logger


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security scheme
security = HTTPBearer()


class JWTManager:
    """JWT token management for authentication."""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            logger.debug("JWT token created successfully")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Failed to create JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            logger.debug("JWT token verified successfully")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError as e:
            logger.warning(f"JWT token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def refresh_token(self, token: str) -> str:
        """Refresh JWT token if it's close to expiry."""
        try:
            payload = self.verify_token(token)
            
            # Remove exp and iat from payload for new token
            payload.pop("exp", None)
            payload.pop("iat", None)
            
            # Create new token
            return self.create_access_token(payload)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not refresh token"
            )


# Global JWT manager instance
jwt_manager = JWTManager()


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to verify JWT token from request headers."""
    try:
        payload = jwt_manager.verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token_payload: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """Get current user from JWT token payload."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    return {
        "user_id": user_id,
        "email": token_payload.get("email"),
        "role": token_payload.get("role", "user"),
        "permissions": token_payload.get("permissions", [])
    }


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])
        if permission not in user_permissions and "admin" not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker


def require_role(role: str):
    """Decorator to require specific role."""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role")
        if user_role != role and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    return role_checker


class APIKeyManager:
    """API key management for service-to-service authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    def create_api_key(
        self, 
        name: str, 
        permissions: list = None, 
        expires_at: Optional[datetime] = None
    ) -> str:
        """Create new API key."""
        import secrets
        
        api_key = f"lol_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions or [],
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0
        }
        
        logger.info(f"API key created: {name}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return metadata."""
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        key_data = self.api_keys[api_key]
        
        # Check if key is expired
        if key_data["expires_at"] and datetime.utcnow() > key_data["expires_at"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired"
            )
        
        # Update usage statistics
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"API key revoked: {api_key[:10]}...")
            return True
        return False


# Global API key manager instance
api_key_manager = APIKeyManager()


def validate_request_size(content_length: Optional[int], max_size: int = 10 * 1024 * 1024):
    """Validate request content size."""
    if content_length and content_length > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request too large. Maximum size: {max_size} bytes"
        )


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    import html
    import re
    
    # HTML escape
    text = html.escape(text)
    
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'<script.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload=',
        r'onerror=',
        r'onclick=',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()