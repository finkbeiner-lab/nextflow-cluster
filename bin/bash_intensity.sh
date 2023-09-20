export exp="20230807-KS1-neuron-optocrispr"
export img_norm_name="identity"
export morphology_channel="GFP-DMD1"
export target_channel="GFP-DMD1"
export chosen_wells="F1"
export chosen_timepoints="T0"
export wells_toggle="include"
export timepoints_toggle="include"

python intensity.py --experiment ${exp} --img_norm_name ${img_norm_name}  \
    --chosen_channels ${morphology_channel} --target_channel ${target_channel} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}