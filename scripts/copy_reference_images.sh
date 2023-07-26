#!/bin/bash

DATA_DIR="$1"
shift 
if [[ $# -lt 2 ]]; then
    DATES=$(ls $DATA_DIR/processed_bags)
else
    DATES="$@"
fi

REF_IMG_DIR="$DATA_DIR/Reference-Images"

for DATE in $DATES; do
    PROC_BAG_DIR="$DATA_DIR/processed_bags/$DATE"
    for insertion_dir in $PROC_BAG_DIR/Insertion*; do
        insertion=$(basename $insertion_dir)
        insertion_num=$( echo $insertion | sed 's/^Insertion//' | xargs printf "%02d" )

        target_lft_img="$REF_IMG_DIR/${insertion_num}-ref_left.png"
        target_rgt_img="$REF_IMG_DIR/${insertion_num}-ref_right.png"
        
        echo Copying over references images for: $insertion_dir

        odir=$insertion_dir/0.0
        mkdir -p $odir
        cp -v "$target_lft_img" "$odir/left.png"
        cp -v "$target_rgt_img" "$odir/right.png"

        echo
    done
done