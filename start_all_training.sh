#!/bin/bash

# turn on bash's job control
set -m

docker run --link cs --name bot2 -p 0.0.0.0:6003:6006 elbotto -n bot2 -rl True -t 0 -m TRAIN &
docker run --link cs --name bot3 -p 0.0.0.0:6004:6006 user/base_rl_bot -n bot3 -rl True -t 1 -m RUN &
docker run --link cs --name bot4 -p 0.0.0.0:6005:6006 user/base_rl_bot -n bot4 -rl True -t 1 -m RUN &

docker run --link cs --name bot1 -p 0.0.0.0:6006:6006 elbotto -n bot1 -rl True -t 0 -m TRAIN -l False