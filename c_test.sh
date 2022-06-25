#!/bin/bash
gcc -Wall -Werror -g "$@" -lcheck -lm -lpthread -lrt -lsubunit -DCHECK
