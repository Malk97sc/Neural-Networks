CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude

SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

SRC = $(SRC_DIR)/matrix.c $(SRC_DIR)/linalg.c
OBJ = $(SRC:.c=.o)

TEST_SRC = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRC:$(TEST_DIR)/%.c=$(BUILD_DIR)/%)

APP = $(BUILD_DIR)/app

all: $(APP)

$(APP): $(OBJ) main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@

test: $(TEST_BINS)

$(BUILD_DIR)/%: $(TEST_DIR)/%.c $(OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(SRC_DIR)/*.o

.PHONY: all test clean