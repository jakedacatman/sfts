# build
from python:3.13.12-alpine as build

workdir /usr/src/app

env PYTHONDONTWRITEBYTECODE 1
env PYTHONUNBUFFERED 1

copy . /usr/src/app

run pip install --upgrade pip
copy ./requirements.txt .requirements.txt
run pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# deploy

from python:3.13.12-alpine

run addgroup --system app
run adduser -S -G app app

env HOME=/home/app
env APP_HOME=/home/app/app
run mkdir $APP_HOME
workdir $APP_HOME

run apk update && apk add netcat-openbsd
copy --from=build /usr/src/app/wheels /wheels
copy --from=build /usr/src/app/requirements.txt .
run pip install --upgrade pip
run pip install --no-cache /wheels/*

copy . $APP_HOME
run chown -R app:app $APP_HOME

user app
