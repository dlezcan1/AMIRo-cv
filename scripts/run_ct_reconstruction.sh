#!/bin/bash

# DATA_DIR=$1; shift;
DATA_DIR="$HOME/data"\
"/Jacynthe-Needle"\
"/2023-05-31"\
"/ct_images/unpacked/results"\
"/2023-05-31"

DEFAULT_OPTS_JSON="$DATA_DIR/ct_reconstruction_options.json"

CT_NPZ_FILES=( $( find $DATA_DIR -type f -name "ct_scan.npz" | sort ) )

for ct_npz_file in ${CT_NPZ_FILES[@]}; do 
    echo "Processing $ct_npz_file ..."
    
    in_dir=$(dirname $ct_npz_file)

    # load the options
    OPTS_JSON="$DEFAULT_OPTS_JSON"
    if [[ -f "$in_dir/ct_reconstruction_options.json" ]]; then
        OPTS_JSON="$in_dir/ct_reconstruction_options.json"
    fi

    python3 ./src/needle_reconstruction_ct.py \
        --debug-images \
        --options-json "$OPTS_JSON" \
        --odir $in_dir \
        $@ \
        $ct_npz_file


    echo
done