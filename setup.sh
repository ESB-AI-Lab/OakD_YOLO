#!/bin/bash

# Link module to samples
cd Samples/DataLoader
ln -s ../../oak_yolo oak_yolo
cd ..
cd ..
cd Samples/DistanceFinder
ln -s ../../oak_yolo oak_yolo

echo "Setup Complete"
