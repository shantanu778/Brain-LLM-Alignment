allstories=("alternateithicatom.hf5" "howtodraw.hf5" "sloth.hf5" "naked.hf5" "souls.hf5" "wheretheressmoke.hf5" "avatar.hf5" "legacy.hf5" "myfirstdaywiththeyankees.hf5" "odetostepfather.hf5" "undertheinfluence.hf5")

subject=('UTS01' 'UTS02' 'UTS03' 'UTS04' 'UTS05' 'UTS06' 'UTS07' 'UTS08' 'UTS09')

for sub in "${subject[@]}"; do
    for stories in "${allstories[@]}"; do
        echo "Processing subject: $sub, ROI: $stories"
        datalad get ds003020/derivative/preprocessed_data/$sub/$stories
    done
done

# Brain-LLM-Alignment/ds003020/derivative/preprocessed_data/UTS01
