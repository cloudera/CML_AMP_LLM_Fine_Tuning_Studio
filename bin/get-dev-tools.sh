#!/bin/bash


# Below is a collection of opinionated development tools that help ease
# development when working remotely on Fine Tuning Studio through a
# CML session. These particularly make CLI environments more friendly
# and install some operating tools to inspect our full stack (i.e. SQLite)


# Get SQLite command line tools (does not yet ship natively within a CML runtime)
cd ~
~/bin/get-sqlite.sh

# Get Atuin for magical shell history.
# https://atuin.sh/
curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh
source ~/.bashrc

# Get ZSH to unleash your terminal.
# https://ohmyz.sh/
bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"
source ~/.bashrc