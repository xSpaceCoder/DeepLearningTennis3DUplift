#!/bin/bash
# ============================================
# Automaticsynthetic dataset creation Script
# All data will be stored in /home/mmc-user/tennisapplication/data/synthetic_dataset
# ============================================
set -e

xvfb-run -a -s "-screen 0 1400x900x24" bash -lc '\
   python -m syntheticdataset.mujocosimulation --num_trajectories 50000 --num_processes 96 --mode groundstroke --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 50000 --num_processes 96 --mode groundstroke --direction close_to_far --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 20000 --num_processes 96 --mode serve --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 20000 --num_processes 96 --mode serve --direction close_to_far --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 3000 --num_processes 96 --mode volley --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 3000 --num_processes 96 --mode volley --direction close_to_far --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 1000 --num_processes 96 --mode smash --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 1000 --num_processes 96 --mode smash --direction close_to_far --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 1000 --num_processes 96 --mode lob --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 1000 --num_processes 96 --mode lob --direction close_to_far --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 3000 --num_processes 96 --mode short --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 3000 --num_processes 96 --mode short --direction close_to_far --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 10000 --num_processes 96 --mode toss --direction far_to_close --folder syntheticdata_03-04 \
   && python -m syntheticdataset.mujocosimulation --num_trajectories 10000 --num_processes 96 --mode toss --direction close_to_far --folder syntheticdata_03-04 \
   '