#!/bin/bash
PROJECT_ROOT=$(pwd)

rm -rf "$PROJECT_ROOT/build"
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" -G Ninja
cmake --build "$PROJECT_ROOT/build"
