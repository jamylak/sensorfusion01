APP := blueprint_canvas
BUILD_DIR := build
SRC_DIR := src
INC_DIR := include

SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))

RAYLIB_DIR ?= /opt/homebrew
PKG_CONFIG_RAYLIB := $(shell pkg-config --cflags --libs raylib 2>/dev/null)

ifeq ($(strip $(PKG_CONFIG_RAYLIB)),)
RAYLIB_CFLAGS := -I$(RAYLIB_DIR)/include
RAYLIB_LIBS := -L$(RAYLIB_DIR)/lib -lraylib
else
RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib)
RAYLIB_LIBS := $(shell pkg-config --libs raylib)
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
PLATFORM_LIBS := -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL
else
PLATFORM_LIBS := -lm -lpthread -ldl -lrt -lX11
endif

CC := cc
CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O2 -I$(INC_DIR) $(RAYLIB_CFLAGS)
LDFLAGS := $(RAYLIB_LIBS) $(PLATFORM_LIBS) -lm

.PHONY: all run clean

all: $(BUILD_DIR)/$(APP)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(APP): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

run: all
	./$(BUILD_DIR)/$(APP)

clean:
	rm -rf $(BUILD_DIR)
