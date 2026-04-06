#!/bin/bash

# Check poetry's python version
required_python_version="3.12"

poetry env use python$required_python_version

mount_point="/mnt/datadrive"
sudo mkdir -p "$mount_point"
sudo chown -R azureuser:azureuser "$mount_point"


poetry lock && poetry install

poetry run pip install flash-attn --no-build-isolation

echo "Setting up data directory..."

poetry_venv_path="${mount_point}/pypoetry-envs/"
mkdir -p $poetry_venv_path

poetry config virtualenvs.path "$poetry_venv_path"
echo "Configured poetry virtualenvs path: $poetry_venv_path"



data_directory="${mount_point}/vision-llm-finetune-data"

# Create directories only if they don't exist

if [ ! -d "$data_directory" ]; then
    sudo mkdir -p "$data_directory"
    echo "Created directory: $data_directory"

    # Set ownership to the current user (assuming 'azureuser')
    sudo chown -R azureuser:azureuser "$data_directory"
else
    echo "Directory already exists: $data_directory"
fi

# Create and set up HuggingFace cache directory
hf_cache_dir="${data_directory}/hf-cache"
mkdir -p "$hf_cache_dir"

# Export HF_HOME to use our cache directory
export HF_HOME="$hf_cache_dir"
echo "Set HF_HOME to: $HF_HOME"



echo "Downloading production prescription dataset..."
poetry run python -m vision_llm_finetune.data.update_training_data


prescription_images_dir="${data_directory}/images/prod-prescriptions"
mkdir -p "$prescription_images_dir"

receipt_images_dir="${data_directory}/images/prod-receipts"
mkdir -p "$receipt_images_dir"

data_directory="${mount_point}/vision-llm-finetune-data"
doclaynet_dataset_dir="${data_directory}/data/DocLayNet"
mkdir -p "$doclaynet_dataset_dir"


cord_dataset_dir="${data_directory}/data/CORD"
mkdir -p "$cord_dataset_dir"


coco_dataset_dir="${data_directory}/data/COCO"
mkdir -p "$coco_dataset_dir"

prescription_dataset_dir="${data_directory}/data/prescription_dataset"
mkdir -p "$prescription_dataset_dir"

# Annotated JSONL files — set PRESCRIPTION_TRAIN_JSONL / PRESCRIPTION_TEST_JSONL
# as environment variables before running this script if the files live elsewhere.
prescription_train_jsonl="${PRESCRIPTION_TRAIN_JSONL:-${data_directory}/annotations/prescriptions-train.jsonl}"
prescription_test_jsonl="${PRESCRIPTION_TEST_JSONL:-${data_directory}/annotations/prescriptions-test.jsonl}"

echo "Downloading production prescription images..."
test_jsonl_arg=""
if [ -f "$prescription_test_jsonl" ]; then
    test_jsonl_arg="--test_jsonl $prescription_test_jsonl"
fi
poetry run python scripts/download_production_images.py \
    --train_jsonl "$prescription_train_jsonl" \
    $test_jsonl_arg \
    --secrets_path ../kera-vision-llm-finetune/secrets.json \
    --images_output_dir "$prescription_images_dir"

echo "Preparing Kera prescription dataset..."
test_arg=""
if [ -f "$prescription_test_jsonl" ]; then
    test_arg="--test_jsonl $prescription_test_jsonl"
fi
poetry run python scripts/prepare_kera_prescription.py \
    --train_jsonl "$prescription_train_jsonl" \
    $test_arg \
    --output_dir "$prescription_dataset_dir" \
    --val_ratio 0.1 \
    --seed 42

echo "Preparing cord dataset..."
poetry run python scripts/prepare_cord.py \
    --output_dir "$cord_dataset_dir" \
    --sample_ratio 1.0 \
    --seed 42

echo "Preparing COCO dataset..."
poetry run python scripts/prepare_coco.py \
    --output_dir "$coco_dataset_dir" \
    --sample_ratio 0.1 \
    --train_samples 10000 \
    --val_samples 2000 \
    --seed 42


echo "Downloading DocLayNet dataset..."
poetry run python scripts/prepare_doclaynet.py \
    --output_dir "$doclaynet_dataset_dir" \
    --sample_ratio 0.1 \
    --seed 42


echo "Merging datasets..."
poetry run python scripts/merge_datasets.py \
    --cord_dir "$cord_dataset_dir" \
    --coco_dir "$coco_dataset_dir" \
    --doclaynet_dir "$doclaynet_dataset_dir" \
    --kera_prescription_dir "$prescription_images_dir" \
    --kera_prescription_splits_dir "$prescription_dataset_dir" \
    --output_dir "${data_directory}/prescription_validation/merged_dataset"
echo "Configuration and setup complete."

