#! /bin/bash

cd /usr/local/hive
sudo chown -R orangepi:orangepi ./*
git config pull.rebase false
git checkout .

echo "--> 获取更新..."

git pull

read -p "--> 更新完成，5秒后自动退出程序" -t 5
