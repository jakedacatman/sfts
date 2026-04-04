# build
from tensorflow/tensorflow:2.21.0 as build

workdir /usr/src/app

env PYTHONDONTWRITEBYTECODE 1
env PYTHONUNBUFFERED 1

copy . /usr/src/app

run pip install --upgrade pip
copy ./requirements.txt .requirements.txt
run pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# deploy

from tensorflow/tensorflow:2.21.0

run groupadd -r runner
run useradd -m -r -g runner runner
run apt-get remove -y python3-blinker #conflicts with newer version installed by flask

env HOME=/home/runner
env APP_HOME=/home/runner/app
run mkdir $APP_HOME
workdir $APP_HOME

copy --from=build /usr/src/app/wheels /wheels
copy --from=build /usr/src/app/requirements.txt .
run pip install --upgrade pip
run pip install --no-cache /wheels/*

copy . $APP_HOME
run chown -R runner:runner $APP_HOME
run ls $APP_HOME

user runner
