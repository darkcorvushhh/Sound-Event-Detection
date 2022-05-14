if [ ! $# -eq 1 ]; then
    echo -e "Usage: $0 <raw_data_dir>"
    exit 1
fi

DATA=$1

mkdir {dev,eval,metadata}

# development set
echo "Preparing development set"
python prepare_wav_csv.py "$DATA/audio/train/weak" "dev/wav.csv"
python extract_feature.py "dev/wav.csv" "dev/feature.h5" --sr 44100 --num_worker 1
ln -s $(realpath "$DATA/label/train/weak.csv") "dev/label.csv"

# evaluation set
echo "Preparing evaluation set"
python prepare_wav_csv.py "$DATA/audio/eval" "eval/wav.csv"
python extract_feature.py "eval/wav.csv" "eval/feature.h5" --sr 44100 --num_worker 1
ln -s $(realpath "$DATA/label/eval/eval.csv") "eval/label.csv"
cp "$DATA/label/class_label_indices.txt" "metadata/class_label_indices.txt"
