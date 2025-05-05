from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.user_service import authenticate_user, create_user, get_user_by_email, get_user_by_username
from pydantic import BaseModel

router = APIRouter()
security = HTTPBasic()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str


def get_current_user(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("user")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не авторизован"
        )
    user = get_user_by_username(db, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден"
        )
    return user

@router.post("/sign-up")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Проверка существования пользователя
    db_user_by_username = get_user_by_username(db, username=user.username)
    if db_user_by_username:
        raise HTTPException(status_code=400, detail="Имя пользователя уже занято")

    db_user_by_email = get_user_by_email(db, email=user.email)
    if db_user_by_email:
        raise HTTPException(status_code=400, detail="Email уже используется")

    # Создаем нового пользователя
    return create_user(db=db, username=user.username, email=user.email, password=user.password)

@router.post("/sign-in", response_model=UserResponse)
def login(user_data: UserLogin, response: Response, request: Request, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль"
        )

    request.session["user"] = user.username
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email
    )

@router.post("/logout")
def logout(request: Request):
    request.session.pop("user", None)
    return {"message": "Выход выполнен успешно"}