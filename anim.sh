#!/bin/bash

set -e

fps=${1:-10}
#remove video if already exists
[ -f "output.mp4" ] && rm output.mp4

#create video
cat $(find . -name "*.png"|sort -V)|ffmpeg -framerate $fps -i - output.mp4 &>/dev/null 
