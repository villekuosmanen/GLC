#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_video.mp4>"
    echo "Example: $0 myvideo.mp4"
    exit 1
fi

INPUT_FILE="$1"
SEGMENT_DURATION=5

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed or not in PATH"
    exit 1
fi

# Get video duration in seconds
DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$INPUT_FILE")
DURATION=${DURATION%.*}  # Remove decimal part

# Calculate number of full segments
FULL_SEGMENTS=$((DURATION / SEGMENT_DURATION))
REMAINDER=$((DURATION % SEGMENT_DURATION))

# Get filename without extension for output naming
BASENAME=$(basename "$INPUT_FILE" .mp4)
OUTPUT_DIR="${BASENAME}_segments"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Input file: $INPUT_FILE"
echo "Duration: ${DURATION} seconds"
echo "Creating $FULL_SEGMENTS full segments of $SEGMENT_DURATION seconds each"
if [ $REMAINDER -gt 0 ]; then
    echo "Plus 1 remaining segment of $REMAINDER seconds"
fi
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create full 5-second segments
for ((i=0; i<FULL_SEGMENTS; i++)); do
    START_TIME=$((i * SEGMENT_DURATION))
    END_TIME=$(((i+1) * SEGMENT_DURATION))
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_t${START_TIME}_t${END_TIME}.mp4"
    
    echo "Creating segment $((i+1))/$FULL_SEGMENTS: ${OUTPUT_FILE}"
    
    ffmpeg -i "$INPUT_FILE" -ss $START_TIME -t $SEGMENT_DURATION -c copy -avoid_negative_ts make_zero "$OUTPUT_FILE" -y -loglevel error
    
    if [ $? -eq 0 ]; then
        echo "✓ Segment $((i+1)) created successfully"
    else
        echo "✗ Error creating segment $((i+1))"
    fi
done

# Create remaining segment if there's leftover time
if [ $REMAINDER -gt 0 ]; then
    START_TIME=$((FULL_SEGMENTS * SEGMENT_DURATION))
    END_TIME=$DURATION
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_t${START_TIME}_t${END_TIME}.mp4"
    
    echo "Creating final segment: ${OUTPUT_FILE} (${REMAINDER} seconds)"
    
    ffmpeg -i "$INPUT_FILE" -ss $START_TIME -c copy -avoid_negative_ts make_zero "$OUTPUT_FILE" -y -loglevel error
    
    if [ $? -eq 0 ]; then
        echo "✓ Final segment created successfully"
    else
        echo "✗ Error creating final segment"
    fi
fi

echo ""
echo "Video splitting complete!"
echo "Segments saved in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"