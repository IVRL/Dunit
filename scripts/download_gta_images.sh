#! /bin/bash

base_gta="/sinergia/gta5"
base_www="https://download.visinf.tu-darmstadt.de/data/from_games/data"

mkdir -p "${base_gta}"
mkdir -p "${base_gta}/zips"

for i in {01..10}
do
    img_file="${base_www}/${i}_images.zip"
    # Download Images
    if ! [ -e "${base_gta}/zips/${i}_images.zip" ];
    then
        echo "Starting download of images: ${i}"
        wget "${img_file}" --no-check-certificate -P "${base_gta}/zips"
    fi
    if [ -e "${base_gta}/images" ];
    then
        echo "Images folder already exists skipping"
    else
        unzip "${base_gta}/zips/${i}_images.zip"
    fi

done

if ! [ -e "${base_gta}/zips/read_mapping.zip" ]
then
    wget "https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip" --no-check-certificate -P "${base_gta}/zips"
fi
unzip "${base_gta}/zips/read_mapping.zip" -d ${base_gta}
