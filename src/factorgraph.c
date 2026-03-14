#include "blueprint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Color path_color;
    Color node_color;
    Color edge_color;
} TrackPalette;

static DVec2 dvec2(double x, double y) {
    DVec2 v = {x, y};
    return v;
}

static const TrackPalette *palette_for_vehicle(int vehicle_id) {
    static const TrackPalette palettes[] = {
        {{110, 202, 255, 180}, {168, 224, 255, 255}, {96, 172, 238, 230}},
        {{132, 236, 176, 180}, {178, 246, 204, 255}, {108, 206, 148, 230}},
        {{255, 184, 108, 180}, {255, 218, 166, 255}, {244, 160, 92, 230}},
        {{216, 152, 255, 180}, {228, 194, 255, 255}, {188, 132, 255, 230}}
    };
    static const TrackPalette neutral = {{124, 132, 148, 150}, {196, 204, 214, 255}, {150, 158, 174, 210}};
    if (vehicle_id < 0) {
        return &neutral;
    }
    return &palettes[vehicle_id % (int)(sizeof(palettes) / sizeof(palettes[0]))];
}

static bool matrix_copy_into(Matrix *dst, const Matrix *src) {
    if (dst == NULL || src == NULL || dst->rows != src->rows || dst->cols != src->cols) {
        return false;
    }
    memcpy(dst->data, src->data, sizeof(*dst->data) * (size_t)src->rows * (size_t)src->cols);
    return true;
}

static bool vector_copy_into(Vector *dst, const Vector *src) {
    if (dst == NULL || src == NULL || dst->size != src->size) {
        return false;
    }
    memcpy(dst->data, src->data, sizeof(*dst->data) * (size_t)src->size);
    return true;
}

static const FactorNode *find_node(const FactorGraph *graph, int id) {
    if (graph == NULL) {
        return NULL;
    }
    if (id >= 0 && id < graph->node_count && graph->nodes[id].id == id) {
        return &graph->nodes[id];
    }
    for (int i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].id == id) {
            return &graph->nodes[i];
        }
    }
    return NULL;
}

static int vehicle_for_node(int node_id, const VehicleTrack *tracks, int track_count) {
    for (int i = 0; i < track_count; ++i) {
        if (node_id >= tracks[i].start_node && node_id <= tracks[i].end_node) {
            return tracks[i].vehicle_id;
        }
    }
    return -1;
}

static void draw_state_vector_panel(const BlueprintEngine *engine, const FactorNode *node, DVec2 origin, Color accent) {
    blueprint_draw_vector_visual(engine, &node->state, origin, 12.0f, false, NULL, accent);
}

static void draw_covariance_panel(const BlueprintEngine *engine, const FactorNode *node, DVec2 origin, Color accent) {
    if (node->covariance.rows < 2 || node->covariance.cols < 2) {
        return;
    }
    blueprint_draw_matrix_heatmap(engine, &node->covariance, origin, 12.0f, false, -1, -1, -1, -1, NULL);
    blueprint_draw_matrix_grid(engine, origin, node->covariance.rows, node->covariance.cols, 12.0, Fade(accent, 0.8f), 0.8f);
}

static void draw_factor_node(const BlueprintEngine *engine, const FactorNode *node, int vehicle_id) {
    const TrackPalette *palette = palette_for_vehicle(vehicle_id);
    DVec2 center = dvec2(node->world_position.x, node->world_position.y);
    if (!blueprint_world_point_visible(engine, center, 80.0)) {
        return;
    }

    if (node->covariance.rows >= 2 && node->covariance.cols >= 2) {
        CovarianceData cov = {
            matrix_get(&node->covariance, 0, 0),
            matrix_get(&node->covariance, 0, 1),
            matrix_get(&node->covariance, 1, 1),
            1.8
        };
        blueprint_draw_covariance_ellipse(engine, center, &cov, Fade(palette->path_color, 0.75f));
    }

    Vector2 p = blueprint_world_to_screen(engine, center);
    DrawCircleV(p, 3.5f, palette->node_color);

    DVec2 state_origin = dvec2(center.x + 18.0, center.y - 32.0);
    DVec2 cov_origin = dvec2(center.x + 18.0, center.y + 6.0);
    draw_state_vector_panel(engine, node, state_origin, palette->node_color);
    draw_covariance_panel(engine, node, cov_origin, palette->node_color);
}

static void draw_information_panel(const BlueprintEngine *engine, const FactorEdge *edge, DVec2 origin, Color accent) {
    blueprint_draw_matrix_heatmap(engine, &edge->information_matrix, origin, 10.0f, false, -1, -1, -1, -1, NULL);
    blueprint_draw_matrix_grid(engine, origin, edge->information_matrix.rows, edge->information_matrix.cols, 10.0, Fade(accent, 0.8f), 0.8f);
}

static void draw_residual_vector(const BlueprintEngine *engine, const FactorEdge *edge, DVec2 origin, Color accent) {
    blueprint_draw_vector_visual(engine, &edge->residual, origin, 10.0f, false, NULL, accent);
}

static void draw_factor_edge(const BlueprintEngine *engine, const FactorGraph *graph, const FactorEdge *edge, const VehicleTrack *tracks, int track_count) {
    const FactorNode *from = find_node(graph, edge->from_node);
    const FactorNode *to = find_node(graph, edge->to_node);
    if (from == NULL || to == NULL) {
        return;
    }

    DVec2 a = dvec2(from->world_position.x, from->world_position.y);
    DVec2 b = dvec2(to->world_position.x, to->world_position.y);
    if (!blueprint_world_segment_visible(engine, a, b, 120.0)) {
        return;
    }

    int vehicle_id = vehicle_for_node(edge->from_node, tracks, track_count);
    const TrackPalette *palette = palette_for_vehicle(vehicle_id);
    double info_trace = 0.0;
    int diag_count = edge->information_matrix.rows < edge->information_matrix.cols ? edge->information_matrix.rows : edge->information_matrix.cols;
    for (int i = 0; i < diag_count; ++i) {
        info_trace += matrix_get(&edge->information_matrix, i, i);
    }
    float thickness = 1.2f + (float)fmin(info_trace * 0.012, 2.8);
    blueprint_draw_directed_edge(engine, a, b, thickness, palette->edge_color, true);

    DVec2 mid = dvec2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
    draw_residual_vector(engine, edge, dvec2(mid.x + 12.0, mid.y - 20.0), palette->edge_color);
    draw_information_panel(engine, edge, dvec2(mid.x + 12.0, mid.y + 14.0), palette->edge_color);

    if (edge->measurement_model.rows > 0 && edge->measurement_model.cols > 0) {
        blueprint_draw_matrix_heatmap(engine, &edge->measurement_model, dvec2(mid.x - 40.0, mid.y - 28.0), 8.0f, false, -1, -1, -1, -1, NULL);
    }
}

bool factor_graph_init(FactorGraph *graph) {
    if (graph == NULL) {
        return false;
    }
    memset(graph, 0, sizeof(*graph));
    return true;
}

void factor_graph_free(FactorGraph *graph) {
    if (graph == NULL) {
        return;
    }
    for (int i = 0; i < graph->node_count; ++i) {
        vector_free_storage(&graph->nodes[i].state);
        matrix_free_storage(&graph->nodes[i].covariance);
    }
    for (int i = 0; i < graph->edge_count; ++i) {
        matrix_free_storage(&graph->edges[i].measurement_model);
        matrix_free_storage(&graph->edges[i].information_matrix);
        vector_free_storage(&graph->edges[i].residual);
    }
    memset(graph, 0, sizeof(*graph));
}

FactorNode *factor_graph_add_node(FactorGraph *graph, int id, const Vector *state, const Matrix *covariance, Vector2 world_position) {
    if (graph == NULL || state == NULL || covariance == NULL || graph->node_count >= MAX_FACTOR_NODES) {
        return NULL;
    }
    FactorNode *node = &graph->nodes[graph->node_count];
    memset(node, 0, sizeof(*node));
    if (!vector_init_storage(&node->state, state->size) ||
        !matrix_init_storage(&node->covariance, covariance->rows, covariance->cols)) {
        vector_free_storage(&node->state);
        matrix_free_storage(&node->covariance);
        return NULL;
    }
    node->id = id;
    node->world_position = world_position;
    vector_copy_into(&node->state, state);
    matrix_copy_into(&node->covariance, covariance);
    graph->node_count++;
    return node;
}

FactorEdge *factor_graph_add_edge(FactorGraph *graph, int from_node, int to_node, const Matrix *measurement_model, const Matrix *information_matrix, const Vector *residual) {
    if (graph == NULL || measurement_model == NULL || information_matrix == NULL || residual == NULL || graph->edge_count >= MAX_FACTOR_EDGES) {
        return NULL;
    }
    FactorEdge *edge = &graph->edges[graph->edge_count];
    memset(edge, 0, sizeof(*edge));
    if (!matrix_init_storage(&edge->measurement_model, measurement_model->rows, measurement_model->cols) ||
        !matrix_init_storage(&edge->information_matrix, information_matrix->rows, information_matrix->cols) ||
        !vector_init_storage(&edge->residual, residual->size)) {
        matrix_free_storage(&edge->measurement_model);
        matrix_free_storage(&edge->information_matrix);
        vector_free_storage(&edge->residual);
        return NULL;
    }
    edge->from_node = from_node;
    edge->to_node = to_node;
    matrix_copy_into(&edge->measurement_model, measurement_model);
    matrix_copy_into(&edge->information_matrix, information_matrix);
    vector_copy_into(&edge->residual, residual);
    graph->edge_count++;
    return edge;
}

bool factor_graph_add_loop_closure(FactorGraph *graph, int from_node, int to_node, const Matrix *measurement_model, const Matrix *information_matrix, const Vector *residual) {
    return factor_graph_add_edge(graph, from_node, to_node, measurement_model, information_matrix, residual) != NULL;
}

void blueprint_draw_large_factor_graph(const BlueprintEngine *engine, const FactorGraph *graph, const VehicleTrack *tracks, int track_count, const char *title) {
    if (engine == NULL || graph == NULL) {
        return;
    }

    if (title != NULL && graph->node_count > 0) {
        const FactorNode *root = &graph->nodes[0];
        Vector2 p = blueprint_world_to_screen(engine, dvec2(root->world_position.x, root->world_position.y - 120.0f));
        DrawText(title, (int)p.x, (int)p.y, 16, (Color){228, 236, 244, 255});
    }

    for (int i = 0; i < track_count; ++i) {
        if (tracks[i].start_node < 0 || tracks[i].end_node <= tracks[i].start_node) {
            continue;
        }
        const TrackPalette *palette = palette_for_vehicle(tracks[i].vehicle_id);
        for (int node_id = tracks[i].start_node + 1; node_id <= tracks[i].end_node && node_id < graph->node_count; ++node_id) {
            const FactorNode *prev = find_node(graph, node_id - 1);
            const FactorNode *cur = find_node(graph, node_id);
            if (prev == NULL || cur == NULL) {
                continue;
            }
            DVec2 a = dvec2(prev->world_position.x, prev->world_position.y);
            DVec2 b = dvec2(cur->world_position.x, cur->world_position.y);
            if (!blueprint_world_segment_visible(engine, a, b, 60.0)) {
                continue;
            }
            Vector2 sa = blueprint_world_to_screen(engine, a);
            Vector2 sb = blueprint_world_to_screen(engine, b);
            DrawLineEx(sa, sb, 2.2f, palette->path_color);
        }
    }

    for (int i = 0; i < graph->edge_count; ++i) {
        draw_factor_edge(engine, graph, &graph->edges[i], tracks, track_count);
    }

    for (int i = 0; i < graph->node_count; ++i) {
        int vehicle_id = vehicle_for_node(graph->nodes[i].id, tracks, track_count);
        draw_factor_node(engine, &graph->nodes[i], vehicle_id);
    }
}
