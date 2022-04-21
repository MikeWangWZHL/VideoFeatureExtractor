VIDEO_PATH="/shared/nas/data/m1/wangz3/Shared_Datasets/VL/WebVid/train_subset_1_percent_video"
DATASET_NAME="tmp"
CSV_SAVE_PATH="./input_csv/$DATASET_NAME"
FEATURE_PATH="./extracted_features/$DATASET_NAME"
PICKLE_PATH="./feature_pickles"
PICKLE_NAME="$DATASET_NAME.features.pickle"

echo "generating input csv..."
python preprocess_generate_csv.py --csv=input.csv --video_root_path $VIDEO_PATH --feature_root_path $FEATURE_PATH --csv_save_path $CSV_SAVE_PATH

echo "extract features..."
python extract.py --csv=$CSV_SAVE_PATH/input.csv --type=s3dg --batch_size=64 --num_decoding_thread=4

echo "convert to pickle..."
python convert_video_feature_to_pickle.py --feature_root_path $FEATURE_PATH --pickle_root_path $PICKLE_PATH --pickle_name $PICKLE_NAME

echo "done generating video features at $PICKLE_PATH/$PICKLE_NAME"