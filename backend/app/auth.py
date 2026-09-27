from datetime import datetime, timedelta, timezone
import jwt, hashlib, hmac, os, base64
from sqlalchemy.orm import Session
from .config import SECRET_KEY
from .db import User

ALGORITHM='HS256'

def hash_password(password: str) -> str:
    salt=os.urandom(16)
    digest=hashlib.scrypt(password.encode(),salt=salt,n=2**14,r=8,p=1)
    return 'scrypt$'+base64.b64encode(salt).decode()+'$'+base64.b64encode(digest).decode()

def verify_password(password: str, stored: str) -> bool:
    try:
        _,salt_b64,digest_b64=stored.split('$',2)
        salt=base64.b64decode(salt_b64); expected=base64.b64decode(digest_b64)
        actual=hashlib.scrypt(password.encode(),salt=salt,n=2**14,r=8,p=1)
        return hmac.compare_digest(actual,expected)
    except Exception:
        return False

def create_token(user_id: int) -> str:
    exp=datetime.now(timezone.utc)+timedelta(hours=8)
    return jwt.encode({'sub':str(user_id),'exp':exp},SECRET_KEY,algorithm=ALGORITHM)

def get_user_from_token(db: Session, token: str) -> User|None:
    try:
        payload=jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        return db.get(User,int(payload['sub']))
    except Exception:
        return None
