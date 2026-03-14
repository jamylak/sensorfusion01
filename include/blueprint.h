#ifndef BLUEPRINT_H
#define BLUEPRINT_H

#include <stdbool.h>
#include <stddef.h>

#include "raylib.h"

typedef struct {
    double x;
    double y;
} DVec2;

typedef enum {
    BLUEPRINT_LAYER_GRID = 0,
    BLUEPRINT_LAYER_GEOMETRY,
    BLUEPRINT_LAYER_SIGNALS,
    BLUEPRINT_LAYER_MATH,
    BLUEPRINT_LAYER_DEBUG,
    BLUEPRINT_LAYER_COUNT
} BlueprintLayer;

typedef struct BlueprintNode {
    char name[64];
    Vector2 world_position;
    void (*draw)(Camera2D cam);
    DVec2 precise_world_position;
    void *user_data;
    BlueprintLayer layer;
    bool visible;
} BlueprintNode;

typedef struct {
    double target_x;
    double target_y;
    double target_goal_x;
    double target_goal_y;
    double zoom;
    double zoom_goal;
    float viewport_width;
    float viewport_height;
    float top_bar_height;
} BlueprintCamera;

typedef struct {
    BlueprintNode *nodes;
    size_t count;
    size_t capacity;
    BlueprintCamera camera;
    int active_page;
    bool paused;
    double time_seconds;
    double signal_phase;
} BlueprintEngine;

typedef struct {
    DVec2 from;
    DVec2 to;
} BlueprintEdge;

typedef struct {
    int rows;
    int cols;
    float *values;
    double cell_size;
} HeatmapData;

typedef struct {
    int rows;
    int cols;
    double spacing;
    double vector_scale;
} ArrowFieldData;

typedef struct {
    DVec2 *points;
    int point_count;
    BlueprintEdge *edges;
    int edge_count;
    double radius;
} GraphClusterData;

typedef struct {
    int rows_a;
    int cols_a;
    int cols_b;
    double cell_size;
} MatrixMultiplyData;

typedef struct {
    double sigma_xx;
    double sigma_xy;
    double sigma_yy;
    double radius_scale;
} CovarianceData;

void blueprint_engine_init(BlueprintEngine *engine, int width, int height);
void blueprint_engine_shutdown(BlueprintEngine *engine);
void blueprint_engine_reset(BlueprintEngine *engine, int width, int height);
void blueprint_engine_set_viewport(BlueprintEngine *engine, int width, int height);
void blueprint_engine_update(BlueprintEngine *engine, float dt);
void blueprint_engine_draw(BlueprintEngine *engine);
void blueprint_engine_add_node(BlueprintEngine *engine, const BlueprintNode *node);

void blueprint_init_demo(BlueprintEngine *engine);

Camera2D blueprint_camera_snapshot(const BlueprintEngine *engine);
DVec2 blueprint_screen_to_world(const BlueprintEngine *engine, Vector2 screen);
Vector2 blueprint_world_to_screen(const BlueprintEngine *engine, DVec2 world);
float blueprint_world_length_to_screen(const BlueprintEngine *engine, double length);
DVec2 blueprint_node_origin(void);
const BlueprintEngine *blueprint_active_engine(void);
const BlueprintNode *blueprint_active_node(void);

void blueprint_draw_world_grid(const BlueprintEngine *engine);
void blueprint_draw_arrow(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color);
void blueprint_draw_signal_arrow(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color, double phase_offset);
void blueprint_draw_directed_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color, bool curved);
void blueprint_draw_matrix_grid(const BlueprintEngine *engine, DVec2 origin, int rows, int cols, double cell_size, Color line_color, float thickness);
void blueprint_draw_heatmap(const BlueprintEngine *engine, DVec2 origin, const HeatmapData *heatmap);
void blueprint_draw_equation_block(const BlueprintEngine *engine, DVec2 origin, const char *title, const char *lines[], int line_count, Color color);
void blueprint_draw_dense_cluster(const BlueprintEngine *engine, const GraphClusterData *cluster, DVec2 origin, Color node_color, Color edge_color);
void blueprint_draw_covariance_ellipse(const BlueprintEngine *engine, DVec2 origin, const CovarianceData *cov, Color color);

#endif
