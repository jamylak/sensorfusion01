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

typedef struct {
    bool active;
    char text[256];
    Vector2 screen_position;
} HoverTooltip;

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

static bool matrix_hover_cell(const BlueprintEngine *engine, DVec2 origin, int rows, int cols, double cell_size, int *out_row, int *out_col) {
    DVec2 mouse = blueprint_screen_to_world(engine, GetMousePosition());
    if (mouse.x < origin.x || mouse.y < origin.y ||
        mouse.x >= origin.x + cols * cell_size || mouse.y >= origin.y + rows * cell_size) {
        return false;
    }
    *out_col = (int)((mouse.x - origin.x) / cell_size);
    *out_row = (int)((mouse.y - origin.y) / cell_size);
    return *out_row >= 0 && *out_row < rows && *out_col >= 0 && *out_col < cols;
}

static void set_tooltip(HoverTooltip *tooltip, const char *text) {
    if (tooltip == NULL || tooltip->active || text == NULL) {
        return;
    }
    tooltip->active = true;
    tooltip->screen_position = GetMousePosition();
    strncpy(tooltip->text, text, sizeof(tooltip->text) - 1);
}

static void draw_tooltip_panel(const HoverTooltip *tooltip) {
    if (tooltip == NULL || !tooltip->active) {
        return;
    }
    int width = MeasureText(tooltip->text, 13) + 18;
    int x = (int)tooltip->screen_position.x + 14;
    int y = (int)tooltip->screen_position.y + 14;
    DrawRectangle(x, y, width, 24, Fade((Color){10, 14, 20, 255}, 0.95f));
    DrawRectangleLines(x, y, width, 24, (Color){112, 132, 156, 255});
    DrawText(tooltip->text, x + 8, y + 6, 13, (Color){230, 236, 244, 255});
}

static void draw_hoverable_matrix(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, Color accent, const char *title, const char *const *row_labels, const char *const *col_labels, const char *tooltip_prefix, HoverTooltip *tooltip) {
    int hover_row = -1;
    int hover_col = -1;
    matrix_hover_cell(engine, origin, matrix->rows, matrix->cols, cell_size, &hover_row, &hover_col);
    blueprint_draw_matrix_heatmap(engine, matrix, origin, cell_size, engine->camera.zoom >= 0.34, -1, -1, hover_row, hover_col, title);
    blueprint_draw_matrix_grid(engine, origin, matrix->rows, matrix->cols, cell_size, Fade(accent, 0.75f), 0.9f);
    if (hover_row >= 0 && hover_col >= 0) {
        char text[256];
        const char *row_name = row_labels != NULL ? row_labels[hover_row] : "row";
        const char *col_name = col_labels != NULL ? col_labels[hover_col] : "col";
        snprintf(text, sizeof(text), "%s %s,%s = %.3f", tooltip_prefix, row_name, col_name, matrix_get(matrix, hover_row, hover_col));
        set_tooltip(tooltip, text);
    }
}

static void draw_hoverable_vector(const BlueprintEngine *engine, const Vector *vector, DVec2 origin, float cell_size, Color accent, const char *title, const char *const *row_labels, const char *tooltip_prefix, HoverTooltip *tooltip) {
    Matrix wrapper = {vector->size, 1, vector->data};
    static const char *col_label[] = {"value"};
    draw_hoverable_matrix(engine, &wrapper, origin, cell_size, accent, title, row_labels, col_label, tooltip_prefix, tooltip);
}

static void vehicle_bounds(const FactorGraph *graph, const VehicleTrack *tracks, int track_count, int vehicle_id, DVec2 *out_min, DVec2 *out_max) {
    bool found = false;
    for (int i = 0; i < graph->node_count; ++i) {
        int node_vehicle = vehicle_for_node(graph->nodes[i].id, tracks, track_count);
        if (node_vehicle != vehicle_id) {
            continue;
        }
        DVec2 p = dvec2(graph->nodes[i].world_position.x, graph->nodes[i].world_position.y);
        if (!found) {
            *out_min = p;
            *out_max = p;
            found = true;
            continue;
        }
        if (p.x < out_min->x) out_min->x = p.x;
        if (p.y < out_min->y) out_min->y = p.y;
        if (p.x > out_max->x) out_max->x = p.x;
        if (p.y > out_max->y) out_max->y = p.y;
    }
    if (!found) {
        *out_min = dvec2(0.0, 0.0);
        *out_max = dvec2(0.0, 0.0);
    }
}

static float point_segment_distance(Vector2 p, Vector2 a, Vector2 b) {
    float abx = b.x - a.x;
    float aby = b.y - a.y;
    float apx = p.x - a.x;
    float apy = p.y - a.y;
    float denom = abx * abx + aby * aby;
    float t = denom > 1e-6f ? (apx * abx + apy * aby) / denom : 0.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float cx = a.x + abx * t;
    float cy = a.y + aby * t;
    float dx = p.x - cx;
    float dy = p.y - cy;
    return sqrtf(dx * dx + dy * dy);
}

static void draw_factor_node(const BlueprintEngine *engine, const FactorNode *node, int vehicle_id, int highlighted_vehicle, HoverTooltip *tooltip) {
    static const char *state_labels[] = {"px", "py", "vx", "vy"};
    static const char *cov_labels[] = {"px", "py"};
    const TrackPalette *palette = palette_for_vehicle(vehicle_id);
    DVec2 center = dvec2(node->world_position.x, node->world_position.y);
    if (!blueprint_world_point_visible(engine, center, 80.0)) {
        return;
    }

    Color node_color = highlighted_vehicle >= 0 && vehicle_id != highlighted_vehicle ? Fade(palette->node_color, 0.35f) : palette->node_color;
    Color path_color = highlighted_vehicle >= 0 && vehicle_id != highlighted_vehicle ? Fade(palette->path_color, 0.25f) : Fade(palette->path_color, 0.82f);
    if (node->covariance.rows >= 2 && node->covariance.cols >= 2) {
        CovarianceData cov = {
            matrix_get(&node->covariance, 0, 0),
            matrix_get(&node->covariance, 0, 1),
            matrix_get(&node->covariance, 1, 1),
            1.8
        };
        blueprint_draw_covariance_ellipse(engine, center, &cov, path_color);
    }

    Vector2 p = blueprint_world_to_screen(engine, center);
    DrawCircleV(p, highlighted_vehicle == vehicle_id ? 4.8f : 3.5f, node_color);

    if (engine->camera.zoom >= 0.16) {
        DVec2 state_origin = dvec2(center.x + 18.0, center.y - 32.0);
        DVec2 cov_origin = dvec2(center.x + 18.0, center.y + 6.0);
        draw_hoverable_vector(engine, &node->state, state_origin, 12.0f, node_color, "state", state_labels, "state", tooltip);
        draw_hoverable_matrix(engine, &node->covariance, cov_origin, 12.0f, node_color, "cov", cov_labels, cov_labels, "cov", tooltip);
    }
}

static void draw_factor_edge(const BlueprintEngine *engine, const FactorGraph *graph, const FactorEdge *edge, const VehicleTrack *tracks, int track_count, int highlighted_vehicle, HoverTooltip *tooltip) {
    static const char *state_labels[] = {"px", "py", "vx", "vy"};
    static const char *meas_labels[] = {"mx", "my"};
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
    int target_vehicle_id = vehicle_for_node(edge->to_node, tracks, track_count);
    bool local_chain = vehicle_id >= 0 && vehicle_id == target_vehicle_id && abs(edge->to_node - edge->from_node) == 1;
    if (local_chain && engine->camera.zoom < 0.14) {
        return;
    }
    const TrackPalette *palette = palette_for_vehicle(vehicle_id);
    double info_trace = 0.0;
    int diag_count = edge->information_matrix.rows < edge->information_matrix.cols ? edge->information_matrix.rows : edge->information_matrix.cols;
    for (int i = 0; i < diag_count; ++i) {
        info_trace += matrix_get(&edge->information_matrix, i, i);
    }
    float thickness = 1.2f + (float)fmin(info_trace * 0.012, 2.8);
    Color edge_color = highlighted_vehicle >= 0 && vehicle_id != highlighted_vehicle ? Fade(palette->edge_color, 0.22f) : palette->edge_color;
    blueprint_draw_directed_edge(engine, a, b, thickness, edge_color, !local_chain);

    if (engine->camera.zoom >= 0.18) {
        DVec2 mid = dvec2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
        draw_hoverable_vector(engine, &edge->residual, dvec2(mid.x + 12.0, mid.y - 20.0), 10.0f, edge_color, "r", meas_labels, "residual", tooltip);
        draw_hoverable_matrix(engine, &edge->information_matrix, dvec2(mid.x + 12.0, mid.y + 14.0), 10.0f, edge_color, "Omega", meas_labels, meas_labels, "info", tooltip);
        if (edge->measurement_model.rows > 0 && edge->measurement_model.cols > 0) {
            draw_hoverable_matrix(engine, &edge->measurement_model, dvec2(mid.x - 40.0, mid.y - 28.0), 8.0f, edge_color, "H", meas_labels, state_labels, "model", tooltip);
        }
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
    HoverTooltip tooltip = {0};
    int highlighted_vehicle = -1;
    if (engine == NULL || graph == NULL) {
        return;
    }

    if (title != NULL && graph->node_count > 0) {
        const FactorNode *root = &graph->nodes[0];
        Vector2 p = blueprint_world_to_screen(engine, dvec2(root->world_position.x, root->world_position.y - 120.0f));
        DrawText(title, (int)p.x, (int)p.y, 16, (Color){228, 236, 244, 255});
        DrawText("lane layout: local chains run left-to-right, GPS anchors sit above, cross-links span lanes", (int)p.x, (int)p.y + 18, 12, (Color){164, 176, 194, 255});
    }

    for (int i = 0; i < track_count; ++i) {
        if (tracks[i].start_node < 0 || tracks[i].end_node < tracks[i].start_node) {
            continue;
        }
        const FactorNode *start = find_node(graph, tracks[i].start_node);
        const FactorNode *end = find_node(graph, tracks[i].end_node);
        if (start == NULL || end == NULL) {
            continue;
        }
        const TrackPalette *palette = palette_for_vehicle(tracks[i].vehicle_id);
        DVec2 lane_a = dvec2(start->world_position.x - 28.0, start->world_position.y);
        DVec2 lane_b = dvec2(end->world_position.x + 28.0, end->world_position.y);
        if (blueprint_world_segment_visible(engine, lane_a, lane_b, 80.0)) {
            Vector2 sa = blueprint_world_to_screen(engine, lane_a);
            Vector2 sb = blueprint_world_to_screen(engine, lane_b);
            Vector2 mouse = GetMousePosition();
            bool lane_near = point_segment_distance(mouse, sa, sb) <= 18.0f;
            DrawLineEx(sa, sb, 1.0f, Fade(palette->path_color, 0.35f));
            Vector2 lp = blueprint_world_to_screen(engine, dvec2(start->world_position.x - 120.0, start->world_position.y - 18.0));
            char label[48];
            snprintf(label, sizeof(label), "vehicle %d", tracks[i].vehicle_id);
            int label_width = MeasureText(label, 13) + 12;
            Rectangle label_rect = {(float)lp.x - 6.0f, (float)lp.y - 3.0f, (float)label_width, 20.0f};
            bool hovered = CheckCollisionPointRec(mouse, label_rect);
            bool show_label = engine->camera.zoom >= 0.11 || hovered || lane_near;
            if (hovered) {
                DVec2 min;
                DVec2 max;
                highlighted_vehicle = tracks[i].vehicle_id;
                vehicle_bounds(graph, tracks, track_count, highlighted_vehicle, &min, &max);
                blueprint_set_minimap_highlight((BlueprintEngine *)engine, dvec2(min.x - 50.0, min.y - 50.0), dvec2(max.x + 50.0, max.y + 50.0), palette->node_color);
                set_tooltip(&tooltip, label);
            }
            if (show_label) {
                DrawRectangleRounded(label_rect, 0.2f, 4, hovered ? Fade(palette->path_color, 0.22f) : Fade((Color){18, 24, 32, 255}, 0.72f));
                DrawRectangleRoundedLinesEx(label_rect, 0.2f, 4, hovered ? 1.8f : 1.0f, hovered ? palette->node_color : Fade(palette->path_color, 0.45f));
                DrawText(label, (int)lp.x, (int)lp.y, 13, palette->node_color);
            }

            if (end != NULL) {
                Vector2 end_p = blueprint_world_to_screen(engine, dvec2(end->world_position.x, end->world_position.y));
                float pulse = 6.0f + 2.0f * sinf((float)engine->time_seconds * 4.0f + (float)i);
                DrawCircleLines((int)end_p.x, (int)end_p.y, pulse, palette->node_color);
            }
        }
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
        draw_factor_edge(engine, graph, &graph->edges[i], tracks, track_count, highlighted_vehicle, &tooltip);
    }

    for (int i = 0; i < graph->node_count; ++i) {
        int vehicle_id = vehicle_for_node(graph->nodes[i].id, tracks, track_count);
        draw_factor_node(engine, &graph->nodes[i], vehicle_id, highlighted_vehicle, &tooltip);
    }

    draw_tooltip_panel(&tooltip);
}
