#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

base_dir=../ds3.6

files_list="
app
"
# Makefile
# Dockerfile
# docker-compose.yml
# requirements.txt
# bin/g.sh
# bin/d.sh
# .devcontainer
# .env
# app/convert.py
# app/main.py

for f in $files_list
do
    d=$(dirname $f)
    if [ "$d" != "." ]; then
        mkdir -p $d
    fi
    cp -pr $base_dir/$f ./$d
done

