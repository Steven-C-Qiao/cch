#!/bin/bash

# Script to untar all tar.gz files from 4DDress_tar to 4DDress directory
# Organizes files by subject ID (clothing type directories are already in the tar files)

# Source and destination directories
SOURCE_DIR="/scratches/kyuban/cq244/datasets/4DDress_tar"
DEST_DIR="/scratches/kyuban/cq244/datasets/4DDress"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist!"
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# Count total tar.gz files for progress tracking
TOTAL_FILES=$(find "$SOURCE_DIR" -name "*.tar.gz" | wc -l)
echo "Found $TOTAL_FILES tar.gz files to extract"

# Counter for progress
CURRENT=0

# Find and extract all tar.gz files
find "$SOURCE_DIR" -name "*.tar.gz" | while read -r tarfile; do
    CURRENT=$((CURRENT + 1))
    filename=$(basename "$tarfile")
    echo "[$CURRENT/$TOTAL_FILES] Processing: $filename"
    
    # Extract subject ID from filename
    # Pattern: '_4D-DRESS_XXXXX_Type.tar.gz'
    if [[ $filename =~ _4D-DRESS_([0-9]{5})_(Inner|Outer)\.tar\.gz ]]; then
        subject_id="${BASH_REMATCH[1]}"
        clothing_type="${BASH_REMATCH[2]}"
        
        # Create subject directory
        subject_dir="$DEST_DIR/$subject_id"
        mkdir -p "$subject_dir"
        
        echo "  Subject ID: $subject_id, Clothing Type: $clothing_type"
        echo "  Extracting to: $subject_dir"
        
        # Extract the tar.gz file to the subject directory
        if tar -xzf "$tarfile" -C "$subject_dir"; then
            echo "  ✓ Successfully extracted: $filename"
        else
            echo "  ✗ Failed to extract: $filename"
        fi
    else
        echo "  ⚠ Warning: Filename '$filename' doesn't match expected pattern '_4D-DRESS_XXXXX_Inner/Outer.tar.gz'"
        echo "  Extracting to root destination directory..."
        
        # Fallback: extract to root destination directory
        if tar -xzf "$tarfile" -C "$DEST_DIR"; then
            echo "  ✓ Successfully extracted: $filename (to root directory)"
        else
            echo "  ✗ Failed to extract: $filename"
        fi
    fi
done

echo "Extraction complete! Files organized in: $DEST_DIR"
echo "Directory structure: $DEST_DIR/SUBJECT_ID/ (with Inner/Outer subdirectories from tar files)"
