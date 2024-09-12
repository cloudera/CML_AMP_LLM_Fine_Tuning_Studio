#!/bin/bash

cd ~
rm -rf sqlite3/
mkdir -p sqlite3/
cd sqlite3/
wget https://www.sqlite.org/2024/sqlite-autoconf-3460100.tar.gz
tar -xzvf sqlite-autoconf-3460100.tar.gz
cd sqlite-autoconf-3460100
./configure
make -j
echo "export PATH=~/sqlite3/sqlite-autoconf-3460100:$PATH" >> ~/.bashrc