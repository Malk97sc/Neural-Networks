#check if a test name was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 test_linalg"
    exit 1
fi

TEST_NAME=$1
EXEC_PATH="./build/$TEST_NAME"
REPORT_DIR="valgrind_report"
REPORT_FILE="$REPORT_DIR/${TEST_NAME}_valgrind.txt"

#check if the executable exists
if [ ! -f "$EXEC_PATH" ]; then
    echo "Error: Executable $EXEC_PATH not found."
    echo "Make sure to run 'make test' first."
    exit 1
fi

mkdir -p "$REPORT_DIR"

echo "Running Valgrind on $TEST_NAME..."
echo "Reports will be saved to $REPORT_FILE"

valgrind --leak-check=full --log-file="$REPORT_FILE" "$EXEC_PATH"

if [ $? -eq 0 ]; then
    echo "Valgrind finished. Checking report for summary..."
    grep -E "ERROR SUMMARY|definitely lost|indirectly lost" "$REPORT_FILE"
else
    echo "Error: Valgrind failed to run."
    exit 1
fi
