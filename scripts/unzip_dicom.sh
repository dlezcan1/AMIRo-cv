#!/bin/bash

IN_DIR="$1"
OUT_DIR="$IN_DIR"
if [[ $# -gt 1 ]]; then
    OUT_DIR=$2
fi

mkdir -p $OUT_DIR
IFS=$'\n'; 
for zip_f in $( find "$IN_DIR" -type f -name '*.dicom.zip' ); do
    sub_d=$( realpath --relative-to="$IN_DIR" "${zip_f%.zip}" )
    unzip_od="$OUT_DIR/$sub_d"
    
    mkdir -p "$unzip_od"
    
    echo "Unzipping: $zip_f -> $unzip_od"
    unzip -q "$zip_f" -d "$unzip_od"

    echo
done
