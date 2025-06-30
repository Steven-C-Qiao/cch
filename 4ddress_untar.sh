#!/bin/bash

# Script to untar all tar.gz files from 4DDress_tar to 4DDress directory

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

# Change to destination directory
cd "$DEST_DIR"

# Count total tar.gz files for progress tracking
TOTAL_FILES=$(find "$SOURCE_DIR" -name "*.tar.gz" | wc -l)
echo "Found $TOTAL_FILES tar.gz files to extract"

# Counter for progress
CURRENT=0

# Find and extract all tar.gz files
find "$SOURCE_DIR" -name "*.tar.gz" | while read -r tarfile; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_FILES] Extracting: $(basename "$tarfile")"
    
    # Extract the tar.gz file
    if tar -xzf "$tarfile"; then
        echo "  ✓ Successfully extracted: $(basename "$tarfile")"
    else
        echo "  ✗ Failed to extract: $(basename "$tarfile")"
    fi
done

echo "Extraction complete! Files extracted to: $DEST_DIR"
