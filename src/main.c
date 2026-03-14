#include "blueprint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    HeatmapData heatmap;
    float values[36];
} HeatmapNode;

static DVec2 dvec2(double x, double y) {
    DVec2 v = {x, y};
    return v;
}

static DVec2 dvec2_min(DVec2 a, DVec2 b) {
    return dvec2(fmin(a.x, b.x), fmin(a.y, b.y));
}

static DVec2 dvec2_max(DVec2 a, DVec2 b) {
    return dvec2(fmax(a.x, b.x), fmax(a.y, b.y));
}

static DVec2 origin_offset(double x, double y) {
    DVec2 base = blueprint_node_origin();
    return dvec2(base.x + x, base.y + y);
}

static const BlueprintNode *active_node(void) {
    return blueprint_active_node();
}

static void node_draw_heatmap(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    const BlueprintNode *node = active_node();
    if (engine == NULL || node == NULL) return;
    HeatmapNode *data = (HeatmapNode *)node->user_data;

    blueprint_draw_heatmap(engine, node->precise_world_position, &data->heatmap);
    const char *lines[] = {
        "H[i,j] = tanh((i-2.5)(j-2.5) / 6)",
        "resolution = 6x6",
        "values rendered directly in cell space"
    };
    blueprint_draw_equation_block(engine, origin_offset(0.0, -90.0), "tensor heatmap", lines, 3, (Color){235, 240, 248, 255});
}

static void node_draw_arrow_field(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    const BlueprintNode *node = active_node();
    if (engine == NULL || node == NULL) return;
    ArrowFieldData *field = (ArrowFieldData *)node->user_data;

    for (int r = 0; r < field->rows; ++r) {
        for (int c = 0; c < field->cols; ++c) {
            double x = (c - field->cols * 0.5) * field->spacing;
            double y = (r - field->rows * 0.5) * field->spacing;
            double vx = sin((x + y) * 0.018) + cos(y * 0.035) * 0.55;
            double vy = cos((x - y) * 0.020) - sin(x * 0.032) * 0.55;
            DVec2 from = origin_offset(x, y);
            DVec2 to = origin_offset(x + vx * field->vector_scale, y + vy * field->vector_scale);
            if (!blueprint_world_segment_visible(engine, from, to, 24.0)) {
                continue;
            }
            blueprint_draw_arrow(engine, from, to, 1.2f, (Color){110, 224, 255, 235});
        }
    }

    const char *lines[] = {
        "v(x,y) = [sin(0.018(x+y)) + 0.55cos(0.035y),",
        "          cos(0.020(x-y)) - 0.55sin(0.032x)]",
        "fully sampled field, no LOD substitution"
    };
    blueprint_draw_equation_block(engine, origin_offset(-280.0, -230.0), "vector field", lines, 3, (Color){215, 236, 248, 255});
}

static void node_draw_graph_cluster(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    const BlueprintNode *node = active_node();
    if (engine == NULL || node == NULL) return;
    GraphClusterData *cluster = (GraphClusterData *)node->user_data;

    blueprint_draw_dense_cluster(engine, cluster, node->precise_world_position, (Color){241, 185, 88, 255}, (Color){198, 140, 58, 120});

    const char *lines[] = {
        "dense directed routing fabric",
        "edge list rendered explicitly",
        "curved and straight edges mixed deterministically"
    };
    blueprint_draw_equation_block(engine, origin_offset(-180.0, -170.0), "graph cluster", lines, 3, (Color){240, 228, 198, 255});
}

static void node_draw_matrix_multiply(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    const BlueprintNode *node = active_node();
    if (engine == NULL || node == NULL) return;
    MatrixMultiplyData *data = (MatrixMultiplyData *)node->user_data;

    DVec2 origin = node->precise_world_position;
    DVec2 a = origin;
    DVec2 b = origin_offset(210.0, 0.0);
    DVec2 c = origin_offset(470.0, 0.0);
    blueprint_draw_matrix_grid(engine, a, data->rows_a, data->cols_a, data->cell_size, (Color){180, 205, 232, 255}, 1.2f);
    blueprint_draw_matrix_grid(engine, b, data->cols_a, data->cols_b, data->cell_size, (Color){180, 205, 232, 255}, 1.2f);
    blueprint_draw_matrix_grid(engine, c, data->rows_a, data->cols_b, data->cell_size, (Color){244, 197, 119, 255}, 1.5f);

    for (int i = 0; i < data->rows_a; ++i) {
        DVec2 from = dvec2(a.x + data->cols_a * data->cell_size + 10.0, a.y + (i + 0.5) * data->cell_size);
        DVec2 to = dvec2(c.x - 20.0, c.y + (i + 0.5) * data->cell_size);
        blueprint_draw_signal_arrow(engine, from, to, 2.2f, (Color){255, 147, 86, 240}, i * 0.12);
    }
    for (int k = 0; k < data->cols_a; ++k) {
        DVec2 from = dvec2(b.x - 18.0, b.y + (k + 0.5) * data->cell_size);
        DVec2 to = dvec2(c.x - 20.0, c.y + (k % data->rows_a + 0.5) * data->cell_size);
        blueprint_draw_signal_arrow(engine, from, to, 1.6f, (Color){104, 228, 178, 220}, 0.3 + k * 0.08);
    }

    const char *lines[] = {
        "C(i,j) = sum_k A(i,k) B(k,j)",
        "signal paths show accumulation into output lattice",
        "all substructure remains visible through zoom"
    };
    blueprint_draw_equation_block(engine, origin_offset(0.0, -90.0), "matrix product", lines, 3, (Color){235, 243, 252, 255});
}

static void node_draw_covariance(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    const BlueprintNode *node = active_node();
    if (engine == NULL || node == NULL) return;
    CovarianceData *cov = (CovarianceData *)node->user_data;

    DVec2 origin = node->precise_world_position;
    blueprint_draw_covariance_ellipse(engine, origin, cov, (Color){182, 116, 255, 255});

    for (int i = -5; i <= 5; ++i) {
        blueprint_draw_arrow(engine, dvec2(origin.x - 180.0, origin.y + i * 30.0), dvec2(origin.x + 180.0, origin.y + i * 30.0), 0.8f, (Color){78, 96, 138, 90});
        blueprint_draw_arrow(engine, dvec2(origin.x + i * 30.0, origin.y + 180.0), dvec2(origin.x + i * 30.0, origin.y - 180.0), 0.8f, (Color){78, 96, 138, 90});
    }

    const char *lines[] = {
        "Sigma = [[9.0, 4.5], [4.5, 3.0]]",
        "principal axes from eig(Sigma)",
        "ellipse is rendered analytically in world space"
    };
    blueprint_draw_equation_block(engine, origin_offset(-200.0, -210.0), "covariance ellipse", lines, 3, (Color){237, 221, 255, 255});
}

static void add_node(BlueprintEngine *engine, const char *name, DVec2 position, BlueprintLayer layer, void *user_data, void (*draw)(Camera2D cam)) {
    BlueprintNode node = {0};
    strncpy(node.name, name, sizeof(node.name) - 1);
    node.world_position = (Vector2){(float)position.x, (float)position.y};
    node.precise_world_position = position;
    node.layer = layer;
    node.user_data = user_data;
    node.draw = draw;
    node.visible = true;
    blueprint_engine_add_node(engine, &node);
}

static void set_node_bounds(BlueprintEngine *engine, size_t index, DVec2 min, DVec2 max) {
    engine->nodes[index].bounds_min = min;
    engine->nodes[index].bounds_max = max;
}

static GraphClusterData make_cluster(void) {
    GraphClusterData cluster = {0};
    cluster.point_count = 180;
    cluster.edge_count = 6200;
    cluster.radius = 3.0;
    cluster.points = calloc((size_t)cluster.point_count, sizeof(*cluster.points));
    cluster.edges = calloc((size_t)cluster.edge_count, sizeof(*cluster.edges));
    if (cluster.points == NULL || cluster.edges == NULL) {
        fprintf(stderr, "failed to allocate cluster data\n");
        exit(1);
    }

    for (int i = 0; i < cluster.point_count; ++i) {
        double ring = 55.0 + (i % 12) * 14.0;
        double angle = (double)i * 0.27;
        cluster.points[i] = dvec2(cos(angle) * ring + cos(angle * 2.0) * 20.0,
                                  sin(angle) * ring + sin(angle * 3.0) * 16.0);
    }

    for (int i = 0; i < cluster.edge_count; ++i) {
        int from = (i * 5 + 3) % cluster.point_count;
        int to = (i * 11 + 17) % cluster.point_count;
        cluster.edges[i].from = cluster.points[from];
        cluster.edges[i].to = cluster.points[to];
    }
    return cluster;
}

void blueprint_init_demo(BlueprintEngine *engine) {
    HeatmapNode *heatmap = calloc(1, sizeof(*heatmap));
    ArrowFieldData *field = calloc(1, sizeof(*field));
    GraphClusterData *cluster = calloc(1, sizeof(*cluster));
    MatrixMultiplyData *matmul = calloc(1, sizeof(*matmul));
    CovarianceData *cov = calloc(1, sizeof(*cov));
    if (heatmap == NULL || field == NULL || cluster == NULL || matmul == NULL || cov == NULL) {
        fprintf(stderr, "failed to allocate demo nodes\n");
        exit(1);
    }

    heatmap->heatmap.rows = 6;
    heatmap->heatmap.cols = 6;
    heatmap->heatmap.values = heatmap->values;
    heatmap->heatmap.cell_size = 34.0;
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double x = (double)r - 2.5;
            double y = (double)c - 2.5;
            heatmap->values[r * 6 + c] = (float)tanh((x * y) / 3.5);
        }
    }

    field->rows = 90;
    field->cols = 90;
    field->spacing = 18.0;
    field->vector_scale = 10.0;

    *cluster = make_cluster();

    matmul->rows_a = 4;
    matmul->cols_a = 5;
    matmul->cols_b = 4;
    matmul->cell_size = 36.0;

    cov->sigma_xx = 9.0;
    cov->sigma_xy = 4.5;
    cov->sigma_yy = 3.0;
    cov->radius_scale = 42.0;

    add_node(engine, "heatmap", dvec2(-820.0, -420.0), BLUEPRINT_LAYER_MATH, heatmap, node_draw_heatmap);
    set_node_bounds(engine, engine->count - 1, dvec2(-820.0, -510.0), dvec2(-820.0 + 6.0 * 34.0, -420.0 + 6.0 * 34.0));
    add_node(engine, "vector-field", dvec2(420.0, -110.0), BLUEPRINT_LAYER_GEOMETRY, field, node_draw_arrow_field);
    {
        double half_span = field->spacing * field->rows * 0.5 + field->vector_scale + 40.0;
        set_node_bounds(engine, engine->count - 1, dvec2(420.0 - half_span, -110.0 - half_span - 140.0), dvec2(420.0 + half_span, -110.0 + half_span));
    }
    add_node(engine, "graph-cluster", dvec2(-140.0, 620.0), BLUEPRINT_LAYER_SIGNALS, cluster, node_draw_graph_cluster);
    {
        DVec2 min = dvec2(1e9, 1e9);
        DVec2 max = dvec2(-1e9, -1e9);
        for (int i = 0; i < cluster->point_count; ++i) {
            min = dvec2_min(min, cluster->points[i]);
            max = dvec2_max(max, cluster->points[i]);
        }
        min.x += -140.0 - 50.0;
        min.y += 620.0 - 50.0;
        max.x += -140.0 + 50.0;
        max.y += 620.0 + 50.0;
        set_node_bounds(engine, engine->count - 1, min, max);
    }
    add_node(engine, "matmul", dvec2(1220.0, 340.0), BLUEPRINT_LAYER_MATH, matmul, node_draw_matrix_multiply);
    set_node_bounds(engine, engine->count - 1, dvec2(1220.0, 250.0), dvec2(1220.0 + 620.0, 340.0 + 5.0 * 36.0 + 40.0));
    add_node(engine, "covariance", dvec2(-1180.0, 760.0), BLUEPRINT_LAYER_GEOMETRY, cov, node_draw_covariance);
    set_node_bounds(engine, engine->count - 1, dvec2(-1380.0, 550.0), dvec2(-980.0, 960.0));
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1440, 900, "Blueprint Canvas");
    SetExitKey(KEY_NULL);
    SetTargetFPS(144);

    BlueprintEngine engine;
    blueprint_engine_init(&engine, GetScreenWidth(), GetScreenHeight());
    blueprint_init_demo(&engine);

    while (!WindowShouldClose() && !engine.quit_requested) {
        blueprint_engine_update(&engine, GetFrameTime());
        blueprint_engine_draw(&engine);
    }

    blueprint_engine_shutdown(&engine);
    CloseWindow();
    return 0;
}
