#!/bin/bash

# turn on bash's job control
set -m

#docker run --link cs --name bot2 -p 0.0.0.0:6003:6006 elbotto -n bot2 -rl True -t 0 -m TRAIN &
#docker run --link cs --name bot3 -p 0.0.0.0:6004:6006 user/base_rl_bot -n bot3 -rl True -t 1 -m RUN &
#docker run --link cs --name bot4 -p 0.0.0.0:6005:6006 user/base_rl_bot -n bot4 -rl True -t 1 -m RUN &
#docker run --link cs --name bot1 -p 0.0.0.0:6006:6006 elbotto -n bot1 -rl True -t 0 -m TRAIN -l False

#docker run --link cs --name bot3 rulebot -Pmyargs=ws://cs:3000,Bot3 > /dev/null 2>&1 &
#docker run --link cs --name bot4 rulebot -Pmyargs=ws://cs:3000,Bot4 > /dev/null 2>&1 &
docker run --link cs --name bot3 rulebot > /dev/null 2>&1 &
docker run --link cs --name bot4 rulebot > /dev/null 2>&1 &
#docker run --link cs --name bot1 -p 0.0.0.0:6006:6006 elbotto -n b1 -rl True -t 1 -m TRAIN > /dev/null 2>&1 &
#docker run --link cs --name bot2 -p 0.0.0.0:6007:6006 elbotto -n b2 -rl True -t 1 -m TRAIN
docker run --link cs --name bot1 -p 0.0.0.0:6006:6006 user/cnnbot2 -n b1 -rl True -t 1 -m TRAIN > /dev/null 2>&1 &
docker run --link cs --name bot2 -p 0.0.0.0:6007:6006 user/cnnbot2 -n b2 -rl True -t 1 -m TRAIN


# After training
# docker ps -a
# docker commit $CONTAINER_ID user/botXYZ