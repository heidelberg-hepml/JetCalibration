#!/bin/bash

# Base URL without the file name
#BASE_URL="https://cernbox.cern.ch/remote.php/dav/public-files/FBJoSO6Q5QuUouk"
BASE_URL="https://cernbox.cern.ch/remote.php/dav/public-files/KjXmPMRorit6zUD"

cd data_v2

FILE_NAME="Ak10Jet.root"
URL="${BASE_URL}/${FILE_NAME}"

echo "Downloading from ${URL}..."
wget --no-check-certificate "${URL}"


for i in {1..82}
do
    FILE_NAME="Ak10Jet_${i}.root"
    URL="${BASE_URL}/${FILE_NAME}"

    echo "Downloading ${FILE_NAME}..."
    wget --no-check-certificate "${URL}"
done

echo "All files downloaded!"