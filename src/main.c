#include "blueprint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FUSION_HISTORY 12
#define FUSION_EVENTS 128
#define DEBUG_FRAMES 96
#define FUSION_VEHICLES 3
#define GRAPH_STATE_DIM 4
#define GRAPH_MEAS_DIM 2
#define GRAPH_SAMPLES_PER_VEHICLE 260
#define GRAPH_ID_STRIDE 300
#define GRAPH_ANCHOR_BASE_ID 1200

enum {
    CAUSAL_NONE = 0,
    CAUSAL_IMU,
    CAUSAL_GPS,
    CAUSAL_CAMERA
};

typedef struct {
    Matrix *a;
    Matrix *b;
    Matrix *c;
    Matrix *a_transpose;
    Matrix *covariance;
    Matrix *covariance_inverse;
    Matrix *transition;
    Matrix *propagated_covariance;
    Matrix *state_matrix;
    Matrix *transformed_state_matrix;
    Vector *state_vector;
    Vector *transformed_vector;
    TensorHeatmap heatmap_a;
    TensorHeatmap heatmap_b;
    TensorHeatmap heatmap_c;
    MathNode nodes[5];
    int node_count;
    DVec2 scene_min;
    DVec2 scene_max;
    float cell_size;
    float vector_cell_size;
} MathSceneData;

typedef struct {
    TrueVehicle true_vehicle;
    GaussianState prior_state;
    GaussianState state_estimate;
    GaussianState predicted_state;
    GaussianState gps_corrected_state;
    GaussianState fused_state;
    SensorModel gps_sensor;
    SensorModel camera_sensor;
    ImuMeasurement imu_measurement;
    GpsMeasurement gps_measurement;
    CameraMeasurement camera_measurement;
    Residual gps_residual;
    Residual camera_residual;
    Matrix gps_gain;
    Matrix camera_gain;
    Matrix process_noise;
    Matrix transition_jacobian;
    Matrix transition_matrix_view;
    Matrix imu_process_view;
    KalmanInternals imu_internals;
    KalmanInternals gps_internals;
    KalmanInternals camera_internals;
    ExecutionTimeline debugger;
    SimulationFrame frame_history[DEBUG_FRAMES];
    int frame_history_count;
    int current_frame;
    double debugger_accum;
    SimulationFrame debug_frame;
    Matrix debug_hp;
    Matrix debug_hpt;
    Matrix debug_predicted_measurement;
    Matrix debug_correction;
    const Matrix *overlay_a;
    const Matrix *overlay_b;
    const Matrix *overlay_c;
    const Vector *overlay_vector;
    const SensorModel *overlay_sensor;
    Matrix covariance_history[12];
    int covariance_history_count;
    DVec2 gps_innovation_history[128];
    int gps_innovation_count;
    DVec2 camera_innovation_history[128];
    int camera_innovation_count;
    DVec2 gps_graph_points[128];
    double gps_graph_residuals[128];
    int gps_graph_count;
    DVec2 camera_graph_from[128];
    DVec2 camera_graph_to[128];
    double camera_graph_weights[128];
    int camera_graph_count;
    double imu_event_times[128];
    int imu_event_count;
    double gps_event_times[128];
    int gps_event_count;
    double camera_event_times[128];
    int camera_event_count;
    DVec2 true_path[512];
    DVec2 predicted_path[512];
    DVec2 fused_path[512];
    int path_count;
    DVec2 feature_from[10];
    DVec2 feature_to[10];
    int feature_count;
    DVec2 last_true_position;
    double last_theta;
    double last_time;
    double next_gps_time;
    double next_camera_time;
    double last_imu_sample_time;
    double last_gps_sample_time;
    double last_camera_sample_time;
    double last_imu_dt;
    bool gps_event_active;
    bool camera_event_active;
    bool imu_event_active;
    int causal_focus;
    double causal_focus_until;
    FactorGraph factor_graph;
    VehicleTrack vehicle_tracks[MAX_VEHICLE_TRACKS];
    int vehicle_track_count;
    int vehicle_last_node_id[FUSION_VEHICLES];
    int vehicle_sample_count[FUSION_VEHICLES];
    double vehicle_next_sample_time[FUSION_VEHICLES];
    int next_anchor_id;
    bool initialized;
    SensorFusionNode imu_node;
    SensorFusionNode gps_node;
    SensorFusionNode camera_node;
    DVec2 scene_min;
    DVec2 scene_max;
    float cell_size;
} FusionSceneData;

typedef struct MatrixInspector {
    bool active;
    bool cell_active;
    int row;
    int col;
    int timeline_index;
    char title[64];
    char body[320];
    Vector2 screen_anchor;
} MatrixInspector;

static MathSceneData *g_math_scene = NULL;
static FusionSceneData *g_fusion_scene = NULL;

static void destroy_math_scene(MathSceneData *scene);
static void destroy_fusion_scene(FusionSceneData *scene);
static void append_factor_graph_samples(FusionSceneData *scene, double t);
static void draw_math_scene_minimap(const BlueprintEngine *engine, Rectangle map_rect, DVec2 world_min, DVec2 world_max);
static void draw_fusion_scene_minimap(const BlueprintEngine *engine, Rectangle map_rect, DVec2 world_min, DVec2 world_max);
static bool hover_world_rect_screen(const BlueprintEngine *engine, DVec2 origin, Vector2 size);
static void set_matrix_inspector(MatrixInspector *inspector, const char *title, const char *body);
static void draw_matrix_inspector_box(const MatrixInspector *inspector);
static void draw_math_matrix_panel(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, const char *title, const char *description, MatrixInspector *inspector, int selected_row, int selected_col, int focus_row, int focus_col, bool *out_hovered, int *out_row, int *out_col);
static void draw_math_vector_panel(const BlueprintEngine *engine, const Vector *vector, DVec2 origin, float cell_size, const char *title, Color accent, const char *description, MatrixInspector *inspector, int selected_index, bool *out_hovered, int *out_index);
static void maybe_describe_math_node(const BlueprintEngine *engine, const MathNode *node, Vector2 size, const char *description, MatrixInspector *inspector);
static void debugger_step_prediction(void);
static void debugger_step_innovation(void);
static void debugger_step_s_matrix(void);
static void debugger_step_gain(void);
static void debugger_step_state_correction(void);
static void debugger_step_covariance_update(void);
static void debugger_draw_overlay(void);

static DVec2 dvec2(double x, double y) {
    DVec2 v = {x, y};
    return v;
}

static const FactorNode *factor_graph_lookup_node(const FactorGraph *graph, int id) {
    if (graph == NULL) {
        return NULL;
    }
    for (int i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].id == id) {
            return &graph->nodes[i];
        }
    }
    return NULL;
}

static void sync_vector_from_matrix(Vector *vector, const Matrix *matrix) {
    for (int i = 0; i < vector->size; ++i) {
        vector->data[i] = matrix_get(matrix, i, 0);
    }
}

static void copy_matrix_values(Matrix *dst, const Matrix *src) {
    memcpy(dst->data, src->data, sizeof(*dst->data) * (size_t)src->rows * (size_t)src->cols);
}

static void push_point(DVec2 *points, int *count, int capacity, DVec2 point) {
    if (*count < capacity) {
        points[*count] = point;
        (*count)++;
        return;
    }
    memmove(points, points + 1, sizeof(*points) * (size_t)(capacity - 1));
    points[capacity - 1] = point;
}

static void copy_gaussian_state(GaussianState *dst, const GaussianState *src) {
    memcpy(dst->mean.data, src->mean.data, sizeof(*dst->mean.data) * (size_t)src->mean.size);
    memcpy(dst->covariance.data, src->covariance.data, sizeof(*dst->covariance.data) * (size_t)src->covariance.rows * (size_t)src->covariance.cols);
}

static void append_fusion_trajectory(FusionSceneData *scene, DVec2 truth, DVec2 predicted, DVec2 fused) {
    push_point(scene->true_path, &scene->path_count, 512, truth);
    if (scene->path_count > 0) {
        int idx = scene->path_count - 1;
        scene->predicted_path[idx] = predicted;
        scene->fused_path[idx] = fused;
    }
}

static void push_sample2(DVec2 *samples, int *count, int capacity, DVec2 sample) {
    if (*count < capacity) {
        samples[*count] = sample;
        (*count)++;
        return;
    }
    memmove(samples, samples + 1, sizeof(*samples) * (size_t)(capacity - 1));
    samples[capacity - 1] = sample;
}

static void push_time_sample(double *times, int *count, int capacity, double value) {
    if (*count < capacity) {
        times[*count] = value;
        (*count)++;
        return;
    }
    memmove(times, times + 1, sizeof(*times) * (size_t)(capacity - 1));
    times[capacity - 1] = value;
}

static void copy_kalman_internals(KalmanInternals *dst, const Matrix *F, const Matrix *Q, const Matrix *H, const Matrix *R, const Matrix *K, const Matrix *S, const Matrix *P, const Vector *innovation) {
    if (F != NULL) copy_matrix_values(&dst->F, F);
    if (Q != NULL) copy_matrix_values(&dst->Q, Q);
    if (H != NULL) copy_matrix_values(&dst->H, H);
    if (R != NULL) copy_matrix_values(&dst->R, R);
    if (K != NULL) copy_matrix_values(&dst->K, K);
    if (S != NULL) copy_matrix_values(&dst->S, S);
    if (P != NULL) copy_matrix_values(&dst->P, P);
    if (innovation != NULL) memcpy(dst->innovation.data, innovation->data, sizeof(*innovation->data) * (size_t)innovation->size);
}

static void push_covariance_history(FusionSceneData *scene, const Matrix *matrix) {
    if (scene->covariance_history_count < FUSION_HISTORY) {
        copy_matrix_values(&scene->covariance_history[scene->covariance_history_count], matrix);
        scene->covariance_history_count++;
        return;
    }
    for (int i = 1; i < FUSION_HISTORY; ++i) {
        copy_matrix_values(&scene->covariance_history[i - 1], &scene->covariance_history[i]);
    }
    copy_matrix_values(&scene->covariance_history[FUSION_HISTORY - 1], matrix);
}

static void push_graph_point(DVec2 *points, double *weights, int *count, int capacity, DVec2 point, double weight) {
    if (*count < capacity) {
        points[*count] = point;
        weights[*count] = weight;
        (*count)++;
        return;
    }
    memmove(points, points + 1, sizeof(*points) * (size_t)(capacity - 1));
    memmove(weights, weights + 1, sizeof(*weights) * (size_t)(capacity - 1));
    points[capacity - 1] = point;
    weights[capacity - 1] = weight;
}

static void push_graph_edge(DVec2 *from, DVec2 *to, double *weights, int *count, int capacity, DVec2 a, DVec2 b, double w) {
    if (*count < capacity) {
        from[*count] = a;
        to[*count] = b;
        weights[*count] = w;
        (*count)++;
        return;
    }
    memmove(from, from + 1, sizeof(*from) * (size_t)(capacity - 1));
    memmove(to, to + 1, sizeof(*to) * (size_t)(capacity - 1));
    memmove(weights, weights + 1, sizeof(*weights) * (size_t)(capacity - 1));
    from[capacity - 1] = a;
    to[capacity - 1] = b;
    weights[capacity - 1] = w;
}

static void copy_simulation_frame(SimulationFrame *dst, const SimulationFrame *src) {
    copy_gaussian_state(&dst->state, &src->state);
    copy_kalman_internals(&dst->internals, &src->internals.F, &src->internals.Q, &src->internals.H, &src->internals.R, &src->internals.K, &src->internals.S, &src->internals.P, &src->internals.innovation);
}

static void push_debug_frame(FusionSceneData *scene, const GaussianState *state, const KalmanInternals *internals) {
    bool follow_latest = scene->frame_history_count == 0 || scene->current_frame >= scene->frame_history_count - 1;
    if (scene->frame_history_count < DEBUG_FRAMES) {
        copy_gaussian_state(&scene->frame_history[scene->frame_history_count].state, state);
        copy_kalman_internals(&scene->frame_history[scene->frame_history_count].internals, &internals->F, &internals->Q, &internals->H, &internals->R, &internals->K, &internals->S, &internals->P, &internals->innovation);
        scene->frame_history_count++;
    } else {
        for (int i = 1; i < DEBUG_FRAMES; ++i) {
            copy_simulation_frame(&scene->frame_history[i - 1], &scene->frame_history[i]);
        }
        copy_gaussian_state(&scene->frame_history[DEBUG_FRAMES - 1].state, state);
        copy_kalman_internals(&scene->frame_history[DEBUG_FRAMES - 1].internals, &internals->F, &internals->Q, &internals->H, &internals->R, &internals->K, &internals->S, &internals->P, &internals->innovation);
    }
    if (follow_latest) {
        scene->current_frame = scene->frame_history_count - 1;
    }
}

static void evaluate_vehicle_state(int vehicle_id, double t, double *x, double *y, double *vx, double *vy) {
    double phase = vehicle_id * 0.9;
    double base_x = -2100.0 + vehicle_id * 1300.0;
    double base_y = 1560.0 + vehicle_id * 220.0;
    *x = base_x + 260.0 * cos(t * 0.22 + phase) + 54.0 * t;
    *y = base_y + 210.0 * sin(t * 0.27 + phase * 0.6) + 120.0 * cos(t * 0.11 + vehicle_id * 0.4);
    *vx = -57.2 * sin(t * 0.22 + phase) + 54.0;
    *vy = 56.7 * cos(t * 0.27 + phase * 0.6) - 13.2 * sin(t * 0.11 + vehicle_id * 0.4);
}

static Vector2 graph_lane_position(int vehicle_id, int sample_index) {
    float x = -2260.0f + (float)sample_index * 17.0f;
    float y = 1720.0f + (float)vehicle_id * 260.0f;
    return (Vector2){x, y};
}

static Vector2 graph_anchor_position(int vehicle_id, int sample_index) {
    float x = -2260.0f + (float)sample_index * 17.0f;
    float y = 1460.0f + (float)vehicle_id * 48.0f;
    return (Vector2){x, y};
}

static bool graph_add_state_node(FusionSceneData *scene, int vehicle_id, int node_id, Vector2 world_position, double t, double x, double y, double vx, double vy) {
    Vector state = {0};
    Matrix covariance = {0};
    bool ok = false;
    if (!vector_init_storage(&state, GRAPH_STATE_DIM) || !matrix_init_storage(&covariance, 2, 2)) {
        goto cleanup;
    }
    state.data[0] = x;
    state.data[1] = y;
    state.data[2] = vx;
    state.data[3] = vy;

    matrix_set(&covariance, 0, 0, 180.0 + vehicle_id * 30.0 + fabs(sin(t * 0.35 + vehicle_id)) * 110.0);
    matrix_set(&covariance, 0, 1, 26.0 * sin(t * 0.18 + vehicle_id * 0.5));
    matrix_set(&covariance, 1, 0, matrix_get(&covariance, 0, 1));
    matrix_set(&covariance, 1, 1, 150.0 + vehicle_id * 24.0 + fabs(cos(t * 0.28 + vehicle_id * 0.7)) * 90.0);

    ok = factor_graph_add_node(&scene->factor_graph, node_id, &state, &covariance, world_position) != NULL;
cleanup:
    vector_free_storage(&state);
    matrix_free_storage(&covariance);
    return ok;
}

static bool graph_add_measurement_edge(FusionSceneData *scene, int from_id, int to_id, double residual_x, double residual_y, double info_x, double info_y, double coupling) {
    Matrix model = {0};
    Matrix information = {0};
    Vector residual = {0};
    bool ok = false;
    if (!matrix_init_storage(&model, GRAPH_MEAS_DIM, GRAPH_STATE_DIM) ||
        !matrix_init_storage(&information, GRAPH_MEAS_DIM, GRAPH_MEAS_DIM) ||
        !vector_init_storage(&residual, GRAPH_MEAS_DIM)) {
        goto cleanup;
    }

    matrix_set(&model, 0, 0, 1.0);
    matrix_set(&model, 0, 2, 0.08);
    matrix_set(&model, 1, 1, 1.0);
    matrix_set(&model, 1, 3, 0.08);
    matrix_set(&information, 0, 0, info_x);
    matrix_set(&information, 1, 1, info_y);
    matrix_set(&information, 0, 1, coupling);
    matrix_set(&information, 1, 0, coupling);
    residual.data[0] = residual_x;
    residual.data[1] = residual_y;
    ok = factor_graph_add_edge(&scene->factor_graph, from_id, to_id, &model, &information, &residual) != NULL;
cleanup:
    matrix_free_storage(&model);
    matrix_free_storage(&information);
    vector_free_storage(&residual);
    return ok;
}

static void append_factor_graph_samples(FusionSceneData *scene, double t) {
    for (int vehicle = 0; vehicle < FUSION_VEHICLES; ++vehicle) {
        while (scene->vehicle_sample_count[vehicle] < GRAPH_SAMPLES_PER_VEHICLE && t >= scene->vehicle_next_sample_time[vehicle]) {
            int sample_index = scene->vehicle_sample_count[vehicle];
            int node_id = vehicle * GRAPH_ID_STRIDE + sample_index;
            double sample_t = scene->vehicle_next_sample_time[vehicle];
            double x = 0.0;
            double y = 0.0;
            double vx = 0.0;
            double vy = 0.0;
            Vector2 lane_position = graph_lane_position(vehicle, sample_index);
            evaluate_vehicle_state(vehicle, sample_t, &x, &y, &vx, &vy);
            if (!graph_add_state_node(scene, vehicle, node_id, lane_position, sample_t, x, y, vx, vy)) {
                return;
            }

            if (sample_index == 0) {
                scene->vehicle_tracks[vehicle].vehicle_id = vehicle;
                scene->vehicle_tracks[vehicle].start_node = node_id;
            }
            scene->vehicle_tracks[vehicle].end_node = node_id;

            if (scene->vehicle_last_node_id[vehicle] >= 0) {
                int prev_id = scene->vehicle_last_node_id[vehicle];
                graph_add_measurement_edge(scene, prev_id, node_id,
                                           1.8 * sin(sample_t * 0.6 + vehicle),
                                           1.6 * cos(sample_t * 0.5 + vehicle * 0.4),
                                           1.0 + vehicle * 0.1,
                                           1.1 + vehicle * 0.1,
                                           0.08);
                graph_add_measurement_edge(scene, prev_id, node_id,
                                           0.7 * sin(sample_t * 0.9 + vehicle * 0.2),
                                           0.6 * cos(sample_t * 0.85 + vehicle * 0.3),
                                           2.8 + vehicle * 0.2,
                                           2.5 + vehicle * 0.2,
                                           0.12);
            }

            if (sample_index > 0 && sample_index % 6 == 0 && scene->next_anchor_id < MAX_FACTOR_NODES) {
                int anchor_id = scene->next_anchor_id++;
                double gps_x = x + 34.0 * sin(sample_t * 0.33 + vehicle);
                double gps_y = y + 30.0 * cos(sample_t * 0.27 + vehicle * 0.7);
                Vector2 anchor_position = graph_anchor_position(vehicle, sample_index);
                if (graph_add_state_node(scene, -1, anchor_id, anchor_position, sample_t, gps_x, gps_y, 0.0, 0.0)) {
                    graph_add_measurement_edge(scene, node_id, anchor_id,
                                               gps_x - x,
                                               gps_y - y,
                                               0.55,
                                               0.55,
                                               0.03);
                }
            }

            if (sample_index > 18 && sample_index % 24 == 0) {
                int loop_id = vehicle * GRAPH_ID_STRIDE + (sample_index - 18);
                graph_add_measurement_edge(scene, loop_id, node_id,
                                           2.2 * sin(sample_t * 0.21),
                                           -2.0 * cos(sample_t * 0.19),
                                           3.4,
                                           3.1,
                                           0.2);
            }

            if (vehicle > 0 && sample_index > 8 && sample_index % 18 == 0) {
                int peer_vehicle = vehicle - 1;
                int peer_index = scene->vehicle_sample_count[peer_vehicle] > sample_index ? sample_index : scene->vehicle_sample_count[peer_vehicle] - 1;
                if (peer_index >= 0) {
                    int peer_id = peer_vehicle * GRAPH_ID_STRIDE + peer_index;
                    graph_add_measurement_edge(scene, peer_id, node_id,
                                               3.0 * sin(sample_t * 0.24 + vehicle),
                                               2.8 * cos(sample_t * 0.26 + vehicle),
                                               1.7,
                                               1.7,
                                               0.16);
                }
            }

            scene->vehicle_last_node_id[vehicle] = node_id;
            scene->vehicle_sample_count[vehicle]++;
            scene->vehicle_next_sample_time[vehicle] += 0.14 + vehicle * 0.015;
        }
    }
}

static bool math_node_active(const BlueprintEngine *engine, int index) {
    int active = ((int)floor(engine->time_seconds * 1.2)) % g_math_scene->node_count;
    return active == index;
}

static bool fusion_node_active(const BlueprintEngine *engine, int index) {
    int active = ((int)floor(engine->time_seconds * 1.4)) % 3;
    return active == index;
}

static void debugger_apply_frame_from_index(FusionSceneData *scene) {
    if (scene->frame_history_count == 0) {
        return;
    }
    if (scene->current_frame < 0) scene->current_frame = 0;
    if (scene->current_frame >= scene->frame_history_count) scene->current_frame = scene->frame_history_count - 1;
    copy_simulation_frame(&scene->debug_frame, &scene->frame_history[scene->current_frame]);
}

static void debugger_common_reset(void) {
    if (g_fusion_scene == NULL) return;
    debugger_apply_frame_from_index(g_fusion_scene);
    g_fusion_scene->overlay_a = NULL;
    g_fusion_scene->overlay_b = NULL;
    g_fusion_scene->overlay_c = NULL;
    g_fusion_scene->overlay_vector = NULL;
    g_fusion_scene->overlay_sensor = NULL;
}

static void debugger_step_prediction(void) {
    debugger_common_reset();
    if (g_fusion_scene == NULL) return;
    copy_kalman_internals(&g_fusion_scene->debug_frame.internals, &g_fusion_scene->imu_internals.F, &g_fusion_scene->imu_internals.Q, NULL, NULL, NULL, NULL, &g_fusion_scene->predicted_state.covariance, NULL);
    copy_gaussian_state(&g_fusion_scene->debug_frame.state, &g_fusion_scene->predicted_state);
    g_fusion_scene->overlay_a = &g_fusion_scene->imu_internals.F;
    g_fusion_scene->overlay_b = &g_fusion_scene->prior_state.covariance;
    g_fusion_scene->overlay_c = &g_fusion_scene->imu_internals.P;
}

static void debugger_step_innovation(void) {
    debugger_common_reset();
    if (g_fusion_scene == NULL) return;
    copy_gaussian_state(&g_fusion_scene->debug_frame.state, &g_fusion_scene->predicted_state);
    copy_kalman_internals(&g_fusion_scene->debug_frame.internals, NULL, NULL, &g_fusion_scene->gps_internals.H, &g_fusion_scene->gps_internals.R, NULL, NULL, &g_fusion_scene->predicted_state.covariance, &g_fusion_scene->gps_internals.innovation);
    g_fusion_scene->overlay_a = &g_fusion_scene->gps_internals.H;
    g_fusion_scene->overlay_b = &g_fusion_scene->debug_predicted_measurement;
    g_fusion_scene->overlay_vector = &g_fusion_scene->gps_internals.innovation;
    g_fusion_scene->overlay_sensor = &g_fusion_scene->gps_sensor;
}

static void debugger_step_s_matrix(void) {
    debugger_common_reset();
    if (g_fusion_scene == NULL) return;
    g_fusion_scene->overlay_a = &g_fusion_scene->gps_internals.H;
    g_fusion_scene->overlay_b = &g_fusion_scene->debug_hp;
    g_fusion_scene->overlay_c = &g_fusion_scene->gps_internals.S;
    g_fusion_scene->overlay_sensor = &g_fusion_scene->gps_sensor;
}

static void debugger_step_gain(void) {
    debugger_common_reset();
    if (g_fusion_scene == NULL) return;
    g_fusion_scene->overlay_a = &g_fusion_scene->predicted_state.covariance;
    g_fusion_scene->overlay_b = &g_fusion_scene->debug_hpt;
    g_fusion_scene->overlay_c = &g_fusion_scene->gps_internals.K;
    g_fusion_scene->overlay_sensor = &g_fusion_scene->gps_sensor;
}

static void debugger_step_state_correction(void) {
    debugger_common_reset();
    if (g_fusion_scene == NULL) return;
    copy_gaussian_state(&g_fusion_scene->debug_frame.state, &g_fusion_scene->gps_corrected_state);
    g_fusion_scene->overlay_a = &g_fusion_scene->gps_internals.K;
    g_fusion_scene->overlay_b = &g_fusion_scene->debug_correction;
    g_fusion_scene->overlay_vector = &g_fusion_scene->gps_internals.innovation;
    g_fusion_scene->overlay_sensor = &g_fusion_scene->gps_sensor;
}

static void debugger_step_covariance_update(void) {
    debugger_common_reset();
    if (g_fusion_scene == NULL) return;
    copy_gaussian_state(&g_fusion_scene->debug_frame.state, &g_fusion_scene->gps_corrected_state);
    copy_kalman_internals(&g_fusion_scene->debug_frame.internals, NULL, NULL, &g_fusion_scene->gps_internals.H, &g_fusion_scene->gps_internals.R, &g_fusion_scene->gps_internals.K, &g_fusion_scene->gps_internals.S, &g_fusion_scene->gps_corrected_state.covariance, &g_fusion_scene->gps_internals.innovation);
    g_fusion_scene->overlay_a = &g_fusion_scene->gps_internals.K;
    g_fusion_scene->overlay_b = &g_fusion_scene->gps_internals.H;
    g_fusion_scene->overlay_c = &g_fusion_scene->gps_corrected_state.covariance;
}

static void debugger_draw_overlay(void) {
    if (g_fusion_scene == NULL) return;
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine == NULL) return;
    const char *desc = g_fusion_scene->debugger.steps[g_fusion_scene->debugger.current_step].description;
    blueprint_draw_debug_inspector(engine, desc, g_fusion_scene->overlay_a, g_fusion_scene->overlay_b, g_fusion_scene->overlay_c, g_fusion_scene->overlay_vector, g_fusion_scene->overlay_sensor, dvec2(1110.0, -520.0), 18.0f);
}

static void compute_matrix_multiply_node(MathNode *node) { matrix_multiply_into(node->inputA, node->inputB, node->output); }
static void compute_matrix_transpose_node(MathNode *node) { matrix_transpose_into(node->inputA, node->output); }
static void compute_matrix_inverse_node(MathNode *node) { matrix_inverse_2x2_into(node->inputA, node->output); }
static void compute_vector_transform_node(MathNode *node) { matrix_multiply_into(node->inputA, node->inputB, node->output); }
static void compute_covariance_propagation_node(MathNode *node) { matrix_covariance_propagate_into(node->inputA, node->inputB, node->output); }

static void draw_multiply_math_node(MathNode *node) {
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine != NULL) blueprint_draw_math_node_box(engine, node, (Vector2){170.0f, 92.0f}, (Color){246, 178, 92, 255}, math_node_active(engine, 0));
}
static void draw_transpose_math_node(MathNode *node) {
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine != NULL) blueprint_draw_math_node_box(engine, node, (Vector2){150.0f, 84.0f}, (Color){110, 202, 255, 255}, math_node_active(engine, 1));
}
static void draw_inverse_math_node(MathNode *node) {
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine != NULL) blueprint_draw_math_node_box(engine, node, (Vector2){150.0f, 84.0f}, (Color){188, 132, 255, 255}, math_node_active(engine, 2));
}
static void draw_vector_transform_math_node(MathNode *node) {
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine != NULL) blueprint_draw_math_node_box(engine, node, (Vector2){164.0f, 84.0f}, (Color){112, 228, 188, 255}, math_node_active(engine, 3));
}
static void draw_covariance_math_node(MathNode *node) {
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine != NULL) blueprint_draw_math_node_box(engine, node, (Vector2){180.0f, 88.0f}, (Color){244, 126, 172, 255}, math_node_active(engine, 4));
}

static void draw_sensor_node(const BlueprintEngine *engine, const SensorFusionNode *node, DVec2 center, Vector2 size, Color accent, int index, const char *title) {
    blueprint_draw_sensor_fusion_node_box(engine, node, center, size, accent, fusion_node_active(engine, index), title);
}

static void init_math_scene_compute(MathSceneData *scene) {
    for (int i = 0; i < scene->node_count; ++i) {
        scene->nodes[i].compute(&scene->nodes[i]);
    }
    sync_vector_from_matrix(scene->transformed_vector, scene->transformed_state_matrix);
}

static void draw_math_graph(const BlueprintEngine *engine, MathSceneData *scene) {
    DVec2 a_center = dvec2(scene->heatmap_a.world_position.x + scene->cell_size * 1.5, scene->heatmap_a.world_position.y + scene->cell_size * 1.5);
    DVec2 b_center = dvec2(scene->heatmap_b.world_position.x + scene->cell_size * 1.5, scene->heatmap_b.world_position.y + scene->cell_size * 1.5);
    DVec2 c_center = dvec2(scene->heatmap_c.world_position.x + scene->cell_size * 1.5, scene->heatmap_c.world_position.y + scene->cell_size * 1.5);
    DVec2 at_center = dvec2(560.0 + scene->cell_size * 1.5, -360.0 + scene->cell_size * 1.5);
    DVec2 cov_center = dvec2(-1120.0 + scene->cell_size, 300.0 + scene->cell_size);
    DVec2 inv_center = dvec2(-650.0 + scene->cell_size, 300.0 + scene->cell_size);
    DVec2 trans_center = dvec2(-80.0 + scene->cell_size, 300.0 + scene->cell_size);
    DVec2 prop_center = dvec2(360.0 + scene->cell_size, 640.0 + scene->cell_size);
    DVec2 x_center = dvec2(300.0 + scene->vector_cell_size * 0.5, 270.0 + scene->vector_cell_size);
    DVec2 fx_center = dvec2(780.0 + scene->vector_cell_size * 0.5, 270.0 + scene->vector_cell_size);

    blueprint_draw_tensor_flow_edge(engine, a_center, dvec2(scene->nodes[0].position.x - 95.0, scene->nodes[0].position.y - 18.0), (Color){246, 178, 92, 255}, "A", math_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, b_center, dvec2(scene->nodes[0].position.x - 95.0, scene->nodes[0].position.y + 18.0), (Color){246, 178, 92, 255}, "B", math_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, dvec2(scene->nodes[0].position.x + 98.0, scene->nodes[0].position.y), c_center, (Color){246, 178, 92, 255}, "C", math_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, a_center, dvec2(scene->nodes[1].position.x - 88.0, scene->nodes[1].position.y), (Color){110, 202, 255, 255}, "A", math_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, dvec2(scene->nodes[1].position.x + 86.0, scene->nodes[1].position.y), at_center, (Color){110, 202, 255, 255}, "A^T", math_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, cov_center, dvec2(scene->nodes[2].position.x - 84.0, scene->nodes[2].position.y), (Color){188, 132, 255, 255}, "P", math_node_active(engine, 2));
    blueprint_draw_tensor_flow_edge(engine, dvec2(scene->nodes[2].position.x + 84.0, scene->nodes[2].position.y), inv_center, (Color){188, 132, 255, 255}, "P^-1", math_node_active(engine, 2));
    blueprint_draw_tensor_flow_edge(engine, trans_center, dvec2(scene->nodes[3].position.x - 92.0, scene->nodes[3].position.y - 18.0), (Color){112, 228, 188, 255}, "F", math_node_active(engine, 3));
    blueprint_draw_tensor_flow_edge(engine, x_center, dvec2(scene->nodes[3].position.x - 92.0, scene->nodes[3].position.y + 18.0), (Color){112, 228, 188, 255}, "x", math_node_active(engine, 3));
    blueprint_draw_tensor_flow_edge(engine, dvec2(scene->nodes[3].position.x + 96.0, scene->nodes[3].position.y), fx_center, (Color){112, 228, 188, 255}, "Fx", math_node_active(engine, 3));
    blueprint_draw_tensor_flow_edge(engine, trans_center, dvec2(scene->nodes[4].position.x - 104.0, scene->nodes[4].position.y - 18.0), (Color){244, 126, 172, 255}, "F", math_node_active(engine, 4));
    blueprint_draw_tensor_flow_edge(engine, cov_center, dvec2(scene->nodes[4].position.x - 104.0, scene->nodes[4].position.y + 18.0), (Color){244, 126, 172, 255}, "P", math_node_active(engine, 4));
    blueprint_draw_tensor_flow_edge(engine, dvec2(scene->nodes[4].position.x + 106.0, scene->nodes[4].position.y), prop_center, (Color){244, 126, 172, 255}, "F P F^T", math_node_active(engine, 4));
}

static void draw_math_scene_node(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine == NULL || g_math_scene == NULL) return;
    init_math_scene_compute(g_math_scene);
    MatrixInspector inspector = {0};
    DVec2 origin_a = dvec2(g_math_scene->heatmap_a.world_position.x, g_math_scene->heatmap_a.world_position.y);
    DVec2 origin_b = dvec2(g_math_scene->heatmap_b.world_position.x, g_math_scene->heatmap_b.world_position.y);
    DVec2 origin_c = dvec2(g_math_scene->heatmap_c.world_position.x, g_math_scene->heatmap_c.world_position.y);
    DVec2 origin_at = dvec2(560.0, -360.0);
    DVec2 origin_p = dvec2(-1120.0, 300.0);
    DVec2 origin_pinv = dvec2(-650.0, 300.0);
    DVec2 origin_f = dvec2(-80.0, 300.0);
    DVec2 origin_x = dvec2(300.0, 270.0);
    DVec2 origin_fx = dvec2(780.0, 270.0);
    DVec2 origin_pp = dvec2(360.0, 640.0);
    int selected_row = -1;
    int selected_col = -1;
    int transpose_row = -1;
    int transpose_col = -1;
    int selected_vector_index = -1;
    bool hovered = false;
    int hover_row = -1;
    int hover_col = -1;
    int hover_index = -1;

    blueprint_draw_matrix_multiply_visualizer(engine, g_math_scene->a, g_math_scene->b, g_math_scene->c,
                                              origin_a, origin_b, origin_c,
                                              g_math_scene->cell_size, engine->time_seconds);

    draw_math_matrix_panel(engine, g_math_scene->a, origin_a, g_math_scene->cell_size, "A",
                           "Left multiplicand. In practice this stands in for a measurement or feature matrix whose rows are consumed during update equations.",
                           &inspector, -1, -1, -1, -1, &hovered, &hover_row, &hover_col);
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
        transpose_row = hover_col;
        transpose_col = hover_row;
    }
    draw_math_matrix_panel(engine, g_math_scene->b, origin_b, g_math_scene->cell_size, "B",
                           "Right multiplicand. In practice this stands in for state, feature, or Jacobian columns combined with A during update and projection steps.",
                           &inspector, -1, -1, -1, -1, &hovered, &hover_row, &hover_col);
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
    }
    draw_math_matrix_panel(engine, g_math_scene->c, origin_c, g_math_scene->cell_size, "C = A x B",
                           "Product matrix. This is the same primitive used to build predicted measurements, innovation covariance terms, and graph normal-equation blocks.",
                           &inspector, selected_row, selected_col, selected_row, selected_col, &hovered, &hover_row, &hover_col);
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
    }
    draw_math_matrix_panel(engine, g_math_scene->a_transpose, origin_at, g_math_scene->cell_size, "A^T",
                           "Transpose of A. Real filters use this constantly in H^T, F^T, and information-form updates.",
                           &inspector, transpose_row, transpose_col, transpose_row, transpose_col, &hovered, &hover_row, &hover_col);
    if (hovered) {
        transpose_row = hover_row;
        transpose_col = hover_col;
        selected_row = hover_col;
        selected_col = hover_row;
    }

    draw_math_matrix_panel(engine, g_math_scene->covariance, origin_p, g_math_scene->cell_size, "P",
                           "Covariance matrix. This is the estimator uncertainty engineers inspect to see position, velocity, and heading confidence and coupling.",
                           &inspector, selected_row, selected_col, selected_row, selected_col, &hovered, &hover_row, &hover_col);
    blueprint_draw_covariance_matrix_visual(engine, g_math_scene->covariance, origin_p, dvec2(-900.0, 460.0), g_math_scene->cell_size, "P");
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
    }
    draw_math_matrix_panel(engine, g_math_scene->covariance_inverse, origin_pinv, g_math_scene->cell_size, "P^-1",
                           "Precision matrix. In SLAM and factor graphs this is the information weighting that says which directions are trusted more strongly.",
                           &inspector, selected_row, selected_col, selected_row, selected_col, &hovered, &hover_row, &hover_col);
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
    }
    draw_math_matrix_panel(engine, g_math_scene->transition, origin_f, g_math_scene->cell_size, "F",
                           "Linear transition operator. In a real IMU/GPS/vision filter this is the prediction Jacobian that pushes state and covariance forward in time.",
                           &inspector, selected_row, selected_col, selected_row, selected_col, &hovered, &hover_row, &hover_col);
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
    }
    draw_math_vector_panel(engine, g_math_scene->state_vector, origin_x, g_math_scene->vector_cell_size, "x", (Color){124, 228, 196, 255},
                           "Input state vector. Think robot pose, velocity, and orientation before prediction or correction.", &inspector, selected_vector_index, &hovered, &hover_index);
    if (hovered) {
        selected_vector_index = hover_index;
    }
    draw_math_vector_panel(engine, g_math_scene->transformed_vector, origin_fx, g_math_scene->vector_cell_size, "Fx", (Color){246, 168, 96, 255},
                           "Predicted state vector after the motion model. This is what later gets compared against sensor measurements.", &inspector, selected_vector_index, &hovered, &hover_index);
    if (hovered) {
        selected_vector_index = hover_index;
    }
    draw_math_matrix_panel(engine, g_math_scene->propagated_covariance, origin_pp, g_math_scene->cell_size, "P'",
                           "Propagated covariance after applying F P F^T. This is the uncertainty ellipse engineers expect to grow or shear after prediction.",
                           &inspector, selected_row, selected_col, selected_row, selected_col, &hovered, &hover_row, &hover_col);
    blueprint_draw_covariance_matrix_visual(engine, g_math_scene->propagated_covariance, origin_pp, dvec2(590.0, 800.0), g_math_scene->cell_size, "P'");
    if (hovered) {
        selected_row = hover_row;
        selected_col = hover_col;
    }

    for (int i = 0; i < g_math_scene->node_count; ++i) g_math_scene->nodes[i].draw(&g_math_scene->nodes[i]);
    maybe_describe_math_node(engine, &g_math_scene->nodes[0], (Vector2){170.0f, 92.0f}, "Real purpose: the atomic multiply used in Kalman updates, feature projection, and factor-graph assembly.", &inspector);
    maybe_describe_math_node(engine, &g_math_scene->nodes[1], (Vector2){150.0f, 84.0f}, "Real purpose: transposes needed for H^T, F^T, and every covariance/information update path.", &inspector);
    maybe_describe_math_node(engine, &g_math_scene->nodes[2], (Vector2){150.0f, 84.0f}, "Real purpose: converts covariance into information weighting for gating, graph optimization, and confidence reasoning.", &inspector);
    maybe_describe_math_node(engine, &g_math_scene->nodes[3], (Vector2){164.0f, 84.0f}, "Real purpose: state prediction from the motion model before any GPS, camera, or other sensor correction arrives.", &inspector);
    maybe_describe_math_node(engine, &g_math_scene->nodes[4], (Vector2){180.0f, 88.0f}, "Real purpose: uncertainty propagation so engineers can inspect how motion makes confidence grow, rotate, or couple.", &inspector);
    draw_math_graph(engine, g_math_scene);
    const char *lines[] = {
        "Page 1 is the low-level math under page 2's IMU, GPS, camera, and factor-graph estimation pipeline",
        "Each block is a real estimator primitive: prediction Jacobians, state projection, covariance propagation, and information weighting",
        "Use this page to inspect the exact matrix/vector mechanics that later appear as sensors, gains, residuals, and uncertainty on page 2"
    };
    blueprint_draw_equation_block(engine, dvec2(-1220.0, -560.0), "linear algebra layer", lines, 3, (Color){235, 242, 249, 255});
    if (hover_world_rect_screen(engine, dvec2(-1220.0, -560.0), (Vector2){640.0f, 120.0f})) {
        set_matrix_inspector(&inspector, "linear algebra layer", "Practical purpose: this page decomposes the exact algebra behind robotics state estimation so you can inspect how sensors, motion models, and uncertainty updates are actually computed.");
    }
    draw_matrix_inspector_box(&inspector);
}

static void initialize_fusion_scene(FusionSceneData *scene) {
    if (scene->initialized) return;
    scene->state_estimate.mean.data[0] = -1180.0;
    scene->state_estimate.mean.data[1] = -40.0;
    scene->state_estimate.mean.data[2] = 78.0;
    scene->state_estimate.mean.data[3] = 22.0;
    scene->state_estimate.mean.data[4] = 0.18;
    matrix_set(&scene->state_estimate.covariance, 0, 0, 4200.0);
    matrix_set(&scene->state_estimate.covariance, 0, 1, 1200.0);
    matrix_set(&scene->state_estimate.covariance, 1, 0, 1200.0);
    matrix_set(&scene->state_estimate.covariance, 1, 1, 3600.0);
    matrix_set(&scene->state_estimate.covariance, 2, 2, 280.0);
    matrix_set(&scene->state_estimate.covariance, 3, 3, 280.0);
    matrix_set(&scene->state_estimate.covariance, 4, 4, 0.18);

    matrix_set(&scene->gps_sensor.H, 0, 0, 1.0);
    matrix_set(&scene->gps_sensor.H, 1, 1, 1.0);
    matrix_set(&scene->gps_sensor.R, 0, 0, 2600.0);
    matrix_set(&scene->gps_sensor.R, 0, 1, 420.0);
    matrix_set(&scene->gps_sensor.R, 1, 0, 420.0);
    matrix_set(&scene->gps_sensor.R, 1, 1, 2100.0);

    matrix_set(&scene->camera_sensor.H, 0, 0, 1.0);
    matrix_set(&scene->camera_sensor.H, 1, 1, 1.0);
    matrix_set(&scene->camera_sensor.R, 0, 0, 900.0);
    matrix_set(&scene->camera_sensor.R, 1, 1, 900.0);

    matrix_set(&scene->process_noise, 0, 0, 24.0);
    matrix_set(&scene->process_noise, 1, 1, 24.0);
    matrix_set(&scene->process_noise, 2, 2, 36.0);
    matrix_set(&scene->process_noise, 3, 3, 36.0);
    matrix_set(&scene->process_noise, 4, 4, 0.02);
    scene->true_vehicle.position = (Vector2){(float)scene->state_estimate.mean.data[0], (float)scene->state_estimate.mean.data[1]};
    scene->true_vehicle.velocity = (Vector2){(float)scene->state_estimate.mean.data[2], (float)scene->state_estimate.mean.data[3]};
    scene->true_vehicle.heading = scene->state_estimate.mean.data[4];
    scene->last_true_position = dvec2(scene->state_estimate.mean.data[0], scene->state_estimate.mean.data[1]);
    scene->last_theta = scene->state_estimate.mean.data[4];
    scene->next_gps_time = 0.8;
    scene->next_camera_time = 0.18;
    scene->last_imu_sample_time = 0.0;
    scene->last_gps_sample_time = -1.0;
    scene->last_camera_sample_time = -1.0;
    scene->last_imu_dt = 0.0;
    copy_gaussian_state(&scene->prior_state, &scene->state_estimate);
    copy_gaussian_state(&scene->predicted_state, &scene->state_estimate);
    copy_gaussian_state(&scene->gps_corrected_state, &scene->state_estimate);
    copy_gaussian_state(&scene->fused_state, &scene->state_estimate);
    copy_kalman_internals(&scene->imu_internals, &scene->transition_matrix_view, &scene->imu_process_view, NULL, NULL, NULL, NULL, &scene->predicted_state.covariance, NULL);
    scene->initialized = true;
}

static void advance_fusion_scene(FusionSceneData *scene, const BlueprintEngine *engine) {
    initialize_fusion_scene(scene);
    append_factor_graph_samples(scene, engine->time_seconds);
    double t = engine->time_seconds;
    if (scene->last_time == 0.0) {
        scene->last_time = t;
    }
    double dt = t - scene->last_time;
    if (dt < 0.0) dt = 0.0;
    if (dt > 0.06) dt = 0.06;
    scene->last_time = t;
    if (dt <= 0.0) return;

    double true_x = -1180.0 + 160.0 * t + 120.0 * sin(t * 0.42);
    double true_y = -40.0 + 190.0 * sin(t * 0.31) + 70.0 * cos(t * 0.17);
    double true_vx = 160.0 + 50.4 * cos(t * 0.42);
    double true_vy = 58.9 * cos(t * 0.31) - 11.9 * sin(t * 0.17);
    double true_ax = -21.168 * sin(t * 0.42);
    double true_ay = -18.259 * sin(t * 0.31) - 2.023 * cos(t * 0.17);
    double true_theta = atan2(true_vy, true_vx);
    double true_omega = (true_theta - scene->last_theta) / dt;
    scene->last_theta = true_theta;
    scene->true_vehicle.position = (Vector2){(float)true_x, (float)true_y};
    scene->true_vehicle.velocity = (Vector2){(float)true_vx, (float)true_vy};
    scene->true_vehicle.heading = true_theta;
    scene->imu_event_active = true;
    scene->gps_event_active = false;
    scene->camera_event_active = false;
    scene->last_imu_dt = dt;

    double c = cos(scene->state_estimate.mean.data[4]);
    double s = sin(scene->state_estimate.mean.data[4]);
    scene->imu_measurement.accel.x = (float)( c * true_ax + s * true_ay + 0.8 * sin(t * 1.3));
    scene->imu_measurement.accel.y = (float)(-s * true_ax + c * true_ay + 0.6 * cos(t * 0.9));
    scene->imu_measurement.accel.z = 0.0f;
    scene->imu_measurement.gyro.z = (float)(true_omega + 0.02 * sin(t * 0.6));

    copy_gaussian_state(&scene->prior_state, &scene->state_estimate);
    imu_propagation_step(&scene->prior_state, &scene->imu_measurement, dt, &scene->process_noise, &scene->predicted_state, &scene->transition_jacobian);
    copy_matrix_values(&scene->transition_matrix_view, &scene->transition_jacobian);
    copy_matrix_values(&scene->imu_process_view, &scene->process_noise);
    copy_kalman_internals(&scene->imu_internals, &scene->transition_jacobian, &scene->process_noise, NULL, NULL, NULL, NULL, &scene->predicted_state.covariance, NULL);
    push_time_sample(scene->imu_event_times, &scene->imu_event_count, FUSION_EVENTS, t);
    scene->last_imu_sample_time = t;
    copy_gaussian_state(&scene->gps_corrected_state, &scene->predicted_state);
    copy_gaussian_state(&scene->fused_state, &scene->predicted_state);

    if (t >= scene->next_gps_time) {
        scene->gps_measurement.lat = true_x + 34.0 * sin(t * 0.47);
        scene->gps_measurement.lon = true_y + 28.0 * cos(t * 0.38);
        matrix_set(&scene->debug_predicted_measurement, 0, 0, scene->predicted_state.mean.data[0]);
        matrix_set(&scene->debug_predicted_measurement, 1, 0, scene->predicted_state.mean.data[1]);
        matrix_multiply_into(&scene->gps_sensor.H, &scene->predicted_state.covariance, &scene->debug_hp);
        matrix_transpose_into(&scene->gps_sensor.H, &scene->debug_hpt);
        gps_measurement_step(&scene->predicted_state, &scene->gps_sensor, &scene->gps_measurement, &scene->gps_residual, &scene->gps_gain, &scene->gps_corrected_state);
        for (int r = 0; r < scene->debug_correction.rows; ++r) {
            double sum = 0.0;
            for (int c_idx = 0; c_idx < scene->debug_correction.cols && c_idx < scene->gps_residual.innovation.size; ++c_idx) {
                sum += matrix_get(&scene->gps_gain, r, c_idx) * scene->gps_residual.innovation.data[c_idx];
            }
            matrix_set(&scene->debug_correction, r, 0, sum);
        }
        copy_kalman_internals(&scene->gps_internals, NULL, NULL, &scene->gps_sensor.H, &scene->gps_sensor.R, &scene->gps_gain, &scene->gps_residual.S, &scene->gps_corrected_state.covariance, &scene->gps_residual.innovation);
        push_sample2(scene->gps_innovation_history, &scene->gps_innovation_count, FUSION_EVENTS, dvec2(scene->gps_residual.innovation.data[0], scene->gps_residual.innovation.data[1]));
        push_time_sample(scene->gps_event_times, &scene->gps_event_count, FUSION_EVENTS, t);
        push_graph_point(scene->gps_graph_points, scene->gps_graph_residuals, &scene->gps_graph_count, FUSION_EVENTS, dvec2(scene->gps_measurement.lat, scene->gps_measurement.lon), hypot(scene->gps_residual.innovation.data[0], scene->gps_residual.innovation.data[1]));
        scene->gps_event_active = true;
        scene->last_gps_sample_time = t;
        scene->next_gps_time += 0.8;
    }

    if (scene->camera_measurement.pose_delta.rows >= 2) {
        matrix_set(&scene->camera_measurement.pose_delta, 0, 0, (true_x - scene->last_true_position.x) + 6.0 * sin(t * 0.8));
        matrix_set(&scene->camera_measurement.pose_delta, 1, 0, (true_y - scene->last_true_position.y) + 6.0 * cos(t * 0.7));
        if (scene->camera_measurement.pose_delta.rows > 2) {
            matrix_set(&scene->camera_measurement.pose_delta, 2, 0, true_theta - scene->state_estimate.mean.data[4]);
        }
    }
    scene->last_true_position = dvec2(true_x, true_y);
    if (t >= scene->next_camera_time) {
        camera_measurement_step(&scene->gps_corrected_state, &scene->camera_sensor, &scene->camera_measurement, &scene->camera_residual, &scene->camera_gain, &scene->fused_state);
        copy_kalman_internals(&scene->camera_internals, NULL, NULL, &scene->camera_sensor.H, &scene->camera_sensor.R, &scene->camera_gain, &scene->camera_residual.S, &scene->fused_state.covariance, &scene->camera_residual.innovation);
        push_sample2(scene->camera_innovation_history, &scene->camera_innovation_count, FUSION_EVENTS, dvec2(scene->camera_residual.innovation.data[0], scene->camera_residual.innovation.data[1]));
        push_time_sample(scene->camera_event_times, &scene->camera_event_count, FUSION_EVENTS, t);
        {
            DVec2 cam_from = dvec2(scene->gps_corrected_state.mean.data[0], scene->gps_corrected_state.mean.data[1]);
            DVec2 cam_to = dvec2(scene->gps_corrected_state.mean.data[0] + matrix_get(&scene->camera_measurement.pose_delta, 0, 0),
                                 scene->gps_corrected_state.mean.data[1] + matrix_get(&scene->camera_measurement.pose_delta, 1, 0));
            push_graph_edge(scene->camera_graph_from, scene->camera_graph_to, scene->camera_graph_weights, &scene->camera_graph_count, FUSION_EVENTS, cam_from, cam_to, 1.0 / (matrix_get(&scene->camera_sensor.R, 0, 0) + 1e-6));
        }
        scene->camera_event_active = true;
        scene->last_camera_sample_time = t;
        scene->next_camera_time += 0.18;
    }
    copy_gaussian_state(&scene->state_estimate, &scene->fused_state);
    if (scene->camera_event_active) {
        scene->causal_focus = CAUSAL_CAMERA;
        scene->causal_focus_until = t + 0.45;
    } else if (scene->gps_event_active) {
        scene->causal_focus = CAUSAL_GPS;
        scene->causal_focus_until = t + 0.45;
    } else {
        scene->causal_focus = CAUSAL_IMU;
        scene->causal_focus_until = t + 0.28;
    }
    push_covariance_history(scene, &scene->fused_state.covariance);
    push_debug_frame(scene, &scene->gps_corrected_state, &scene->gps_internals);
    append_fusion_trajectory(scene, dvec2(true_x, true_y), dvec2(scene->predicted_state.mean.data[0], scene->predicted_state.mean.data[1]), dvec2(scene->fused_state.mean.data[0], scene->fused_state.mean.data[1]));

    scene->feature_count = 8;
    for (int i = 0; i < scene->feature_count; ++i) {
        double angle = (double)i * 0.75 + t * 0.1;
        double ring = 70.0 + (i % 4) * 18.0;
        scene->feature_from[i] = dvec2(scene->gps_corrected_state.mean.data[0] + cos(angle) * ring,
                                       scene->gps_corrected_state.mean.data[1] + sin(angle) * ring);
        scene->feature_to[i] = dvec2(scene->feature_from[i].x + matrix_get(&scene->camera_measurement.pose_delta, 0, 0) * 0.3,
                                     scene->feature_from[i].y + matrix_get(&scene->camera_measurement.pose_delta, 1, 0) * 0.3);
    }
}

static void draw_velocity_arrow(const BlueprintEngine *engine, const GaussianState *state, Color color, const char *label) {
    DVec2 from = dvec2(state->mean.data[0], state->mean.data[1]);
    DVec2 to = dvec2(from.x + state->mean.data[2] * 0.9, from.y + state->mean.data[3] * 0.9);
    blueprint_draw_arrow(engine, from, to, 1.8f, color);
    if (label != NULL) {
        Vector2 p = blueprint_world_to_screen(engine, to);
        DrawText(label, (int)p.x + 6, (int)p.y - 6, 13, color);
    }
}

static void draw_true_vehicle_marker(const BlueprintEngine *engine, const TrueVehicle *vehicle, Color color) {
    DVec2 center = dvec2(vehicle->position.x, vehicle->position.y);
    Vector2 p = blueprint_world_to_screen(engine, center);
    DrawCircleV(p, 6.0f, color);
    DVec2 nose = dvec2(center.x + cos(vehicle->heading) * 42.0, center.y + sin(vehicle->heading) * 42.0);
    blueprint_draw_arrow(engine, center, nose, 2.1f, color);
}

static void draw_timestamp_label(const BlueprintEngine *engine, DVec2 world, double timestamp, Color color, const char *prefix) {
    Vector2 p = blueprint_world_to_screen(engine, world);
    char label[64];
    snprintf(label, sizeof(label), "%s t=%.2f", prefix, timestamp);
    DrawText(label, (int)p.x + 8, (int)p.y - 10, 12, color);
}

static void draw_causal_chain_overlay(const BlueprintEngine *engine, const FusionSceneData *scene) {
    const char *title = "causal chain";
    const char *steps_imu[] = {
        "true vehicle motion",
        "IMU accel / gyro sample",
        "prediction step",
        "covariance stretches"
    };
    const char *steps_gps[] = {
        "true vehicle motion",
        "GPS position sample",
        "innovation vector",
        "Kalman gain",
        "state correction",
        "covariance shrink"
    };
    const char *steps_cam[] = {
        "true vehicle motion",
        "camera pose delta",
        "feature flow",
        "innovation vector",
        "Kalman gain",
        "state correction"
    };
    const char **steps = steps_imu;
    int step_count = 4;
    Color color = (Color){118, 208, 255, 255};
    if (scene->causal_focus == CAUSAL_GPS) {
        steps = steps_gps;
        step_count = 6;
        color = (Color){248, 176, 102, 255};
    } else if (scene->causal_focus == CAUSAL_CAMERA) {
        steps = steps_cam;
        step_count = 6;
        color = (Color){196, 132, 255, 255};
    }
    blueprint_draw_equation_block(engine, dvec2(1120.0, -520.0), title, steps, step_count, color);
}

static const char *state_axis_label(int index) {
    static const char *labels[] = {"x", "y", "vx", "vy", "theta"};
    if (index < 0 || index >= (int)(sizeof(labels) / sizeof(labels[0]))) {
        return "?";
    }
    return labels[index];
}

static bool hover_matrix_cell_world(const BlueprintEngine *engine, DVec2 origin, int rows, int cols, float cell_size, int *out_row, int *out_col) {
    DVec2 mouse = blueprint_screen_to_world(engine, GetMousePosition());
    if (mouse.x < origin.x || mouse.y < origin.y ||
        mouse.x >= origin.x + cols * cell_size || mouse.y >= origin.y + rows * cell_size) {
        return false;
    }
    *out_col = (int)((mouse.x - origin.x) / cell_size);
    *out_row = (int)((mouse.y - origin.y) / cell_size);
    return *out_row >= 0 && *out_row < rows && *out_col >= 0 && *out_col < cols;
}

static bool hover_matrix_title_screen(const BlueprintEngine *engine, DVec2 origin, float cell_size, const char *title) {
    if (title == NULL) {
        return false;
    }
    Vector2 anchor = blueprint_world_to_screen(engine, dvec2(origin.x, origin.y - cell_size * 0.8));
    int width = MeasureText(title, 16);
    Rectangle rect = {anchor.x - 4.0f, anchor.y - 2.0f, (float)width + 8.0f, 20.0f};
    return CheckCollisionPointRec(GetMousePosition(), rect);
}

static void set_matrix_inspector(MatrixInspector *inspector, const char *title, const char *body) {
    if (inspector == NULL || inspector->active) {
        return;
    }
    inspector->active = true;
    inspector->screen_anchor = GetMousePosition();
    strncpy(inspector->title, title, sizeof(inspector->title) - 1);
    strncpy(inspector->body, body, sizeof(inspector->body) - 1);
}

static void draw_matrix_inspector_box(const MatrixInspector *inspector) {
    if (inspector == NULL || !inspector->active) {
        return;
    }
    int title_w = MeasureText(inspector->title, 15);
    int body_w = MeasureText(inspector->body, 13);
    int width = title_w > body_w ? title_w : body_w;
    width += 20;
    int x = (int)inspector->screen_anchor.x + 18;
    int y = (int)inspector->screen_anchor.y + 18;
    DrawRectangle(x, y, width, 46, Fade((Color){10, 14, 20, 255}, 0.96f));
    DrawRectangleLines(x, y, width, 46, (Color){108, 126, 150, 255});
    DrawText(inspector->title, x + 8, y + 6, 15, (Color){236, 242, 249, 255});
    DrawText(inspector->body, x + 8, y + 24, 13, (Color){188, 202, 220, 255});
}

static void draw_fusion_matrix_panel(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, const char *title, MatrixInspector *inspector, int selected_row, int selected_col, bool *out_hovered_cell, int *out_row, int *out_col) {
    int hover_row = -1;
    int hover_col = -1;
    bool cell_hover = hover_matrix_cell_world(engine, origin, matrix->rows, matrix->cols, cell_size, &hover_row, &hover_col);
    if (out_hovered_cell != NULL) *out_hovered_cell = cell_hover;
    if (out_row != NULL) *out_row = hover_row;
    if (out_col != NULL) *out_col = hover_col;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        char body[320];
        if (strcmp(title, "F") == 0) {
            snprintf(body, sizeof(body), "State transition Jacobian. Each cell maps how one state component influences another during prediction.");
        } else if (strcmp(title, "Q") == 0) {
            snprintf(body, sizeof(body), "Process noise covariance. Each entry models injected uncertainty between propagated state dimensions.");
        } else {
            snprintf(body, sizeof(body), "Matrix panel for %s.", title);
        }
        set_matrix_inspector(inspector, title, body);
    }

    if (cell_hover) {
        char body[320];
        snprintf(body, sizeof(body), "%s[%s,%s] = %.3f", title, state_axis_label(hover_row), state_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col));
        if (strcmp(title, "F") == 0) {
            snprintf(body, sizeof(body), "F[%s,%s] = %.3f. Prediction coupling from %s into %s.", state_axis_label(hover_row), state_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col), state_axis_label(hover_col), state_axis_label(hover_row));
        } else if (strcmp(title, "Q") == 0) {
            snprintf(body, sizeof(body), "Q[%s,%s] = %.3f. Process noise covariance between %s and %s.", state_axis_label(hover_row), state_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col), state_axis_label(hover_row), state_axis_label(hover_col));
        }
        set_matrix_inspector(inspector, title, body);
    }

    blueprint_draw_matrix_heatmap(engine, matrix, origin, cell_size, true,
                                  selected_row, selected_col,
                                  cell_hover ? hover_row : selected_row,
                                  cell_hover ? hover_col : selected_col,
                                  title);
}

static void draw_interactive_covariance_timeline(const BlueprintEngine *engine, const Matrix *history, int history_count, int matrix_dim, DVec2 origin, float cell_size, MatrixInspector *inspector, int selected_row, int selected_col, bool *out_hovered_cell, int *out_row, int *out_col) {
    bool any_hover = false;
    int hover_row = -1;
    int hover_col = -1;
    int hover_index = -1;
    Vector2 title_anchor = blueprint_world_to_screen(engine, dvec2(origin.x, origin.y - cell_size * 2.4));
    if (CheckCollisionPointRec(GetMousePosition(), (Rectangle){title_anchor.x - 4.0f, title_anchor.y - 2.0f, (float)MeasureText("covariance evolution", 15) + 8.0f, 20.0f})) {
        set_matrix_inspector(inspector, "P[...]",
                             "Covariance history. Each P[k] stores uncertainty coupling between state dimensions over time.");
    }
    DrawText("covariance evolution", (int)title_anchor.x, (int)title_anchor.y, 15, (Color){232, 238, 246, 255});

    for (int i = 0; i < history_count; ++i) {
        DVec2 panel_origin = dvec2(origin.x + i * (matrix_dim * cell_size + cell_size * 0.8), origin.y);
        char label[32];
        snprintf(label, sizeof(label), "P[%d]", i);
        int local_row = -1;
        int local_col = -1;
        bool local_hover = hover_matrix_cell_world(engine, panel_origin, matrix_dim, matrix_dim, cell_size, &local_row, &local_col);
        if (hover_matrix_title_screen(engine, panel_origin, cell_size, label)) {
            char body[320];
            snprintf(body, sizeof(body), "P[%d] is the covariance snapshot at timeline step %d.", i, i);
            set_matrix_inspector(inspector, label, body);
        }
        if (local_hover) {
            any_hover = true;
            hover_row = local_row;
            hover_col = local_col;
            hover_index = i;
            char body[320];
            snprintf(body, sizeof(body), "P[%d][%s,%s] = %.3f. Uncertainty coupling at timeline step %d.", i, state_axis_label(local_row), state_axis_label(local_col), matrix_get(&history[i], local_row, local_col), i);
            set_matrix_inspector(inspector, label, body);
        }
        blueprint_draw_matrix_heatmap(engine, &history[i], panel_origin, cell_size, false,
                                      selected_row, selected_col,
                                      local_hover ? local_row : selected_row,
                                      local_hover ? local_col : selected_col,
                                      label);
    }
    if (out_hovered_cell != NULL) *out_hovered_cell = any_hover;
    if (out_row != NULL) *out_row = hover_row;
    if (out_col != NULL) *out_col = hover_col;
    if (inspector != NULL) inspector->timeline_index = hover_index;
}

static const char *measurement_axis_label(int index) {
    static const char *labels[] = {"mx", "my"};
    if (index < 0 || index >= (int)(sizeof(labels) / sizeof(labels[0]))) {
        return "?";
    }
    return labels[index];
}

static void draw_sensor_matrix_panel(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, const char *title, const char *sensor_name, bool measurement_covariance, MatrixInspector *inspector, int selected_row, int selected_col, bool *out_hovered_cell, int *out_row, int *out_col) {
    int hover_row = -1;
    int hover_col = -1;
    bool cell_hover = hover_matrix_cell_world(engine, origin, matrix->rows, matrix->cols, cell_size, &hover_row, &hover_col);
    if (out_hovered_cell != NULL) *out_hovered_cell = cell_hover;
    if (out_row != NULL) *out_row = hover_row;
    if (out_col != NULL) *out_col = hover_col;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        char body[320];
        if (measurement_covariance) {
            snprintf(body, sizeof(body), "%s measurement noise covariance. These entries describe uncertainty and coupling in the %s update.", sensor_name, sensor_name);
        } else {
            snprintf(body, sizeof(body), "%s observation model. Each row maps state dimensions into the %s measurement space.", sensor_name, sensor_name);
        }
        set_matrix_inspector(inspector, title, body);
    }

    if (cell_hover) {
        char body[320];
        if (measurement_covariance) {
            snprintf(body, sizeof(body), "%s[%s,%s] = %.3f. %s noise covariance between measurement axes.", title, measurement_axis_label(hover_row), measurement_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col), sensor_name);
        } else {
            snprintf(body, sizeof(body), "%s[%s,%s] = %.3f. %s maps state %s into measurement %s.", title, measurement_axis_label(hover_row), state_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col), sensor_name, state_axis_label(hover_col), measurement_axis_label(hover_row));
        }
        set_matrix_inspector(inspector, title, body);
    }

    blueprint_draw_matrix_heatmap(engine, matrix, origin, cell_size, true,
                                  selected_row, selected_col,
                                  cell_hover ? hover_row : selected_row,
                                  cell_hover ? hover_col : selected_col,
                                  title);
}

static void draw_gain_matrix_panel(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, const char *title, const char *sensor_name, MatrixInspector *inspector, int selected_row, int selected_col, bool *out_hovered_cell, int *out_row, int *out_col) {
    int hover_row = -1;
    int hover_col = -1;
    bool cell_hover = hover_matrix_cell_world(engine, origin, matrix->rows, matrix->cols, cell_size, &hover_row, &hover_col);
    if (out_hovered_cell != NULL) *out_hovered_cell = cell_hover;
    if (out_row != NULL) *out_row = hover_row;
    if (out_col != NULL) *out_col = hover_col;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        char body[320];
        snprintf(body, sizeof(body), "Kalman gain for %s. Each row says how strongly a state dimension reacts to a measurement residual.", sensor_name);
        set_matrix_inspector(inspector, title, body);
    }
    if (cell_hover) {
        char body[320];
        snprintf(body, sizeof(body), "%s[%s,%s] = %.3f. %s update weight from measurement %s into state %s.", title, state_axis_label(hover_row), measurement_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col), sensor_name, measurement_axis_label(hover_col), state_axis_label(hover_row));
        set_matrix_inspector(inspector, title, body);
    }

    blueprint_draw_matrix_heatmap(engine, matrix, origin, cell_size, true,
                                  selected_row, selected_col,
                                  cell_hover ? hover_row : selected_row,
                                  cell_hover ? hover_col : selected_col,
                                  title);
}

static void draw_residual_matrix_panel(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, const char *title, const char *sensor_name, MatrixInspector *inspector, int selected_row, int selected_col, bool *out_hovered_cell, int *out_row, int *out_col) {
    int hover_row = -1;
    int hover_col = -1;
    bool cell_hover = hover_matrix_cell_world(engine, origin, matrix->rows, matrix->cols, cell_size, &hover_row, &hover_col);
    if (out_hovered_cell != NULL) *out_hovered_cell = cell_hover;
    if (out_row != NULL) *out_row = hover_row;
    if (out_col != NULL) *out_col = hover_col;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        char body[320];
        snprintf(body, sizeof(body), "Innovation covariance for %s. This panel determines expected residual spread before the update is trusted.", sensor_name);
        set_matrix_inspector(inspector, title, body);
    }
    if (cell_hover) {
        char body[320];
        snprintf(body, sizeof(body), "%s[%s,%s] = %.3f. %s innovation covariance between measurement axes.", title, measurement_axis_label(hover_row), measurement_axis_label(hover_col), matrix_get(matrix, hover_row, hover_col), sensor_name);
        set_matrix_inspector(inspector, title, body);
    }

    blueprint_draw_matrix_heatmap(engine, matrix, origin, cell_size, true,
                                  selected_row, selected_col,
                                  cell_hover ? hover_row : selected_row,
                                  cell_hover ? hover_col : selected_col,
                                  title);
}

static void draw_innovation_vector_panel(const BlueprintEngine *engine, const Vector *vector, DVec2 origin, float cell_size, const char *title, Color accent, const char *sensor_name, MatrixInspector *inspector, int selected_index, bool *out_hovered, int *out_index) {
    Matrix wrapper = {vector->size, 1, vector->data};
    int hover_row = -1;
    int hover_col = -1;
    bool hovered = hover_matrix_cell_world(engine, origin, wrapper.rows, wrapper.cols, cell_size, &hover_row, &hover_col);
    if (out_hovered != NULL) *out_hovered = hovered;
    if (out_index != NULL) *out_index = hover_row;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        char body[320];
        snprintf(body, sizeof(body), "%s residual vector. These entries are the measurement-minus-prediction mismatch that drives the %s correction.", sensor_name, sensor_name);
        set_matrix_inspector(inspector, title, body);
    }
    if (hovered) {
        char body[320];
        snprintf(body, sizeof(body), "%s[%s] = %.3f. %s residual along measurement axis %s.", title, measurement_axis_label(hover_row), vector->data[hover_row], sensor_name, measurement_axis_label(hover_row));
        set_matrix_inspector(inspector, title, body);
    }

    blueprint_draw_vector_visual(engine, vector, origin, cell_size, true, title, accent);
    if (selected_index >= 0) {
        blueprint_draw_matrix_heatmap(engine, &wrapper, origin, cell_size, true, selected_index, 0, selected_index, 0, title);
    }
}

static bool hover_world_rect_screen(const BlueprintEngine *engine, DVec2 origin, Vector2 size) {
    Vector2 a = blueprint_world_to_screen(engine, origin);
    Vector2 b = blueprint_world_to_screen(engine, dvec2(origin.x + size.x, origin.y + size.y));
    Rectangle rect = {a.x, a.y, b.x - a.x, b.y - a.y};
    if (rect.width < 0.0f) {
        rect.x += rect.width;
        rect.width = -rect.width;
    }
    if (rect.height < 0.0f) {
        rect.y += rect.height;
        rect.height = -rect.height;
    }
    return CheckCollisionPointRec(GetMousePosition(), rect);
}

static void draw_math_matrix_panel(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, const char *title, const char *description, MatrixInspector *inspector, int selected_row, int selected_col, int focus_row, int focus_col, bool *out_hovered, int *out_row, int *out_col) {
    int hover_row = -1;
    int hover_col = -1;
    bool hovered = hover_matrix_cell_world(engine, origin, matrix->rows, matrix->cols, cell_size, &hover_row, &hover_col);
    if (out_hovered != NULL) *out_hovered = hovered;
    if (out_row != NULL) *out_row = hover_row;
    if (out_col != NULL) *out_col = hover_col;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        set_matrix_inspector(inspector, title, description);
    }
    if (hovered) {
        char body[320];
        snprintf(body, sizeof(body), "%s[%d,%d] = %.3f. %s", title, hover_row, hover_col, matrix_get(matrix, hover_row, hover_col), description);
        set_matrix_inspector(inspector, title, body);
    }

    blueprint_draw_matrix_heatmap(engine, matrix, origin, cell_size, true,
                                  selected_row, selected_col,
                                  hovered ? hover_row : focus_row,
                                  hovered ? hover_col : focus_col,
                                  title);
}

static void draw_math_vector_panel(const BlueprintEngine *engine, const Vector *vector, DVec2 origin, float cell_size, const char *title, Color accent, const char *description, MatrixInspector *inspector, int selected_index, bool *out_hovered, int *out_index) {
    Matrix wrapper = {vector->size, 1, vector->data};
    int hover_row = -1;
    int hover_col = -1;
    bool hovered = hover_matrix_cell_world(engine, origin, wrapper.rows, wrapper.cols, cell_size, &hover_row, &hover_col);
    if (out_hovered != NULL) *out_hovered = hovered;
    if (out_index != NULL) *out_index = hover_row;

    if (hover_matrix_title_screen(engine, origin, cell_size, title)) {
        set_matrix_inspector(inspector, title, description);
    }
    if (hovered) {
        char body[320];
        snprintf(body, sizeof(body), "%s[%d] = %.3f. %s", title, hover_row, vector->data[hover_row], description);
        set_matrix_inspector(inspector, title, body);
    }
    blueprint_draw_vector_visual(engine, vector, origin, cell_size, true, title, accent);
    if (selected_index >= 0) {
        blueprint_draw_matrix_heatmap(engine, &wrapper, origin, cell_size, true, selected_index, 0, selected_index, 0, title);
    }
}

static void maybe_describe_math_node(const BlueprintEngine *engine, const MathNode *node, Vector2 size, const char *description, MatrixInspector *inspector) {
    DVec2 origin = dvec2(node->position.x - size.x * 0.5, node->position.y - size.y * 0.5);
    if (hover_world_rect_screen(engine, origin, size)) {
        set_matrix_inspector(inspector, node->name, description);
    }
}

static void maybe_describe_fusion_node(const BlueprintEngine *engine, DVec2 center, Vector2 size, const char *title, const char *description, MatrixInspector *inspector) {
    DVec2 origin = dvec2(center.x - size.x * 0.5, center.y - size.y * 0.5);
    if (hover_world_rect_screen(engine, origin, size)) {
        set_matrix_inspector(inspector, title, description);
    }
}

static void draw_minimap_rect(Rectangle map_rect, DVec2 world_min, DVec2 world_max, DVec2 min, DVec2 max, Color color) {
    Vector2 a = blueprint_minimap_project(map_rect, world_min, world_max, min);
    Vector2 b = blueprint_minimap_project(map_rect, world_min, world_max, max);
    float x = fminf(a.x, b.x);
    float y = fminf(a.y, b.y);
    float w = fabsf(b.x - a.x);
    float h = fabsf(b.y - a.y);
    if (w < 2.0f) w = 2.0f;
    if (h < 2.0f) h = 2.0f;
    DrawRectangleLinesEx((Rectangle){x, y, w, h}, 1.0f, color);
}

static void draw_minimap_line(Rectangle map_rect, DVec2 world_min, DVec2 world_max, DVec2 a, DVec2 b, float thickness, Color color) {
    Vector2 sa = blueprint_minimap_project(map_rect, world_min, world_max, a);
    Vector2 sb = blueprint_minimap_project(map_rect, world_min, world_max, b);
    DrawLineEx(sa, sb, thickness, color);
}

static void draw_math_scene_minimap(const BlueprintEngine *engine, Rectangle map_rect, DVec2 world_min, DVec2 world_max) {
    (void)engine;
    if (g_math_scene == NULL) {
        return;
    }
    float cs = g_math_scene->cell_size;
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-1180.0, -360.0), dvec2(-1180.0 + 3.0 * cs, -360.0 + 3.0 * cs), (Color){246, 178, 92, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-760.0, -360.0), dvec2(-760.0 + 3.0 * cs, -360.0 + 3.0 * cs), (Color){246, 178, 92, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(20.0, -360.0), dvec2(20.0 + 3.0 * cs, -360.0 + 3.0 * cs), (Color){246, 178, 92, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(560.0, -360.0), dvec2(560.0 + 3.0 * cs, -360.0 + 3.0 * cs), (Color){110, 202, 255, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-1120.0, 300.0), dvec2(-1120.0 + 2.0 * cs, 300.0 + 2.0 * cs), (Color){188, 132, 255, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-650.0, 300.0), dvec2(-650.0 + 2.0 * cs, 300.0 + 2.0 * cs), (Color){188, 132, 255, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-80.0, 300.0), dvec2(-80.0 + 2.0 * cs, 300.0 + 2.0 * cs), (Color){112, 228, 188, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(300.0, 270.0), dvec2(300.0 + g_math_scene->vector_cell_size, 270.0 + 2.0 * g_math_scene->vector_cell_size), (Color){112, 228, 188, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(780.0, 270.0), dvec2(780.0 + g_math_scene->vector_cell_size, 270.0 + 2.0 * g_math_scene->vector_cell_size), (Color){246, 168, 96, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(360.0, 640.0), dvec2(360.0 + 2.0 * cs, 640.0 + 2.0 * cs), (Color){244, 126, 172, 220});
    for (int i = 0; i < g_math_scene->node_count; ++i) {
        DVec2 c = dvec2(g_math_scene->nodes[i].position.x, g_math_scene->nodes[i].position.y);
        Vector2 p = blueprint_minimap_project(map_rect, world_min, world_max, c);
        DrawCircleV(p, 2.5f, (Color){228, 236, 244, 220});
    }
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-1000.0, -180.0), dvec2(-320.0, -170.0), 1.0f, (Color){246, 178, 92, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-580.0, -180.0), dvec2(-320.0, -170.0), 1.0f, (Color){246, 178, 92, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-320.0, -170.0), dvec2(120.0, -180.0), 1.0f, (Color){246, 178, 92, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-20.0, 360.0), dvec2(560.0, 470.0), 1.0f, (Color){112, 228, 188, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-1000.0, 360.0), dvec2(40.0, 760.0), 1.0f, (Color){244, 126, 172, 160});
}

static void draw_fusion_scene_minimap(const BlueprintEngine *engine, Rectangle map_rect, DVec2 world_min, DVec2 world_max) {
    (void)engine;
    if (g_fusion_scene == NULL) {
        return;
    }
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-325.0, -298.0), dvec2(-115.0, -202.0), (Color){118, 208, 255, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(235.0, -298.0), dvec2(445.0, -202.0), (Color){248, 176, 102, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(790.0, -298.0), dvec2(1010.0, -202.0), (Color){196, 132, 255, 220});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-1280.0, 420.0), dvec2(-1280.0 + 5.0 * g_fusion_scene->cell_size, 420.0 + 5.0 * g_fusion_scene->cell_size), (Color){118, 208, 255, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-760.0, 420.0), dvec2(-760.0 + 5.0 * g_fusion_scene->cell_size, 420.0 + 5.0 * g_fusion_scene->cell_size), (Color){244, 170, 116, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-220.0, 420.0), dvec2(-220.0 + 5.0 * g_fusion_scene->cell_size, 420.0 + 3.3 * g_fusion_scene->cell_size), (Color){248, 176, 102, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(340.0, 420.0), dvec2(340.0 + 5.0 * g_fusion_scene->cell_size, 420.0 + 3.3 * g_fusion_scene->cell_size), (Color){196, 132, 255, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(900.0, 420.0), dvec2(900.0 + 5.0 * g_fusion_scene->cell_size, 420.0 + 2.0 * g_fusion_scene->cell_size), (Color){248, 176, 102, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(1320.0, 420.0), dvec2(1320.0 + 5.0 * g_fusion_scene->cell_size, 420.0 + 2.0 * g_fusion_scene->cell_size), (Color){196, 132, 255, 180});
    draw_minimap_rect(map_rect, world_min, world_max, dvec2(-1320.0, 860.0), dvec2(-1320.0 + 12.0 * 5.0 * 18.0f / 5.0, 860.0 + 5.0 * 18.0f), (Color){168, 186, 210, 140});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-900.0, -70.0), dvec2(-340.0, -250.0), 1.0f, (Color){118, 208, 255, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(-100.0, -250.0), dvec2(340.0, -40.0), 1.0f, (Color){118, 208, 255, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(340.0, -40.0), dvec2(450.0, -250.0), 1.0f, (Color){248, 176, 102, 180});
    draw_minimap_line(map_rect, world_min, world_max, dvec2(560.0, -250.0), dvec2(900.0, -250.0), 1.0f, (Color){196, 132, 255, 180});
    if (g_fusion_scene->factor_graph.node_count > 1) {
        for (int i = 0; i < g_fusion_scene->vehicle_track_count; ++i) {
            VehicleTrack track = g_fusion_scene->vehicle_tracks[i];
            if (track.start_node < 0 || track.end_node <= track.start_node) {
                continue;
            }
            for (int node_id = track.start_node + 1; node_id <= track.end_node; ++node_id) {
                const FactorNode *prev = factor_graph_lookup_node(&g_fusion_scene->factor_graph, node_id - 1);
                const FactorNode *cur = factor_graph_lookup_node(&g_fusion_scene->factor_graph, node_id);
                if (prev == NULL || cur == NULL) {
                    continue;
                }
                draw_minimap_line(map_rect, world_min, world_max,
                                  dvec2(prev->world_position.x, prev->world_position.y),
                                  dvec2(cur->world_position.x, cur->world_position.y),
                                  1.0f,
                                  i == 0 ? (Color){110, 202, 255, 180} : (i == 1 ? (Color){132, 236, 176, 180} : (Color){255, 184, 108, 180}));
            }
        }
    }
}

static void debugger_execute_current_step(FusionSceneData *scene) {
    if (scene == NULL || scene->debugger.step_count == 0) return;
    AlgorithmStep *step = &scene->debugger.steps[scene->debugger.current_step];
    if (step->execute != NULL) {
        step->execute();
    }
}

static void handle_debugger_controls(FusionSceneData *scene) {
    if (IsKeyPressed(KEY_P)) {
        scene->debugger.running = !scene->debugger.running;
    }
    if (IsKeyPressed(KEY_N)) {
        scene->debugger.running = false;
        execution_timeline_step_forward(&scene->debugger);
        debugger_execute_current_step(scene);
    }
    if (IsKeyPressed(KEY_B)) {
        scene->debugger.running = false;
        execution_timeline_step_backward(&scene->debugger);
        debugger_execute_current_step(scene);
    }
    if (IsKeyPressed(KEY_U) && scene->frame_history_count > 0) {
        if (scene->current_frame > 0) scene->current_frame--;
        debugger_execute_current_step(scene);
    }
    if (IsKeyPressed(KEY_O) && scene->frame_history_count > 0) {
        if (scene->current_frame < scene->frame_history_count - 1) scene->current_frame++;
        debugger_execute_current_step(scene);
    }
}

static void draw_fusion_scene_node(Camera2D cam) {
    (void)cam;
    const BlueprintEngine *engine = blueprint_active_engine();
    if (engine == NULL || g_fusion_scene == NULL) return;
    advance_fusion_scene(g_fusion_scene, engine);
    handle_debugger_controls(g_fusion_scene);
    g_fusion_scene->debugger_accum += GetFrameTime();
    if (g_fusion_scene->debugger.running && g_fusion_scene->debugger_accum >= 0.6) {
        g_fusion_scene->debugger_accum = 0.0;
        execution_timeline_step_forward(&g_fusion_scene->debugger);
    }
    debugger_execute_current_step(g_fusion_scene);

    DVec2 predicted = dvec2(g_fusion_scene->predicted_state.mean.data[0], g_fusion_scene->predicted_state.mean.data[1]);
    DVec2 gps_meas = dvec2(g_fusion_scene->gps_measurement.lat, g_fusion_scene->gps_measurement.lon);
    DVec2 fused = dvec2(g_fusion_scene->fused_state.mean.data[0], g_fusion_scene->fused_state.mean.data[1]);
    DVec2 gps_corrected = dvec2(g_fusion_scene->gps_corrected_state.mean.data[0], g_fusion_scene->gps_corrected_state.mean.data[1]);
    DVec2 true_vehicle = dvec2(g_fusion_scene->true_vehicle.position.x, g_fusion_scene->true_vehicle.position.y);
    DVec2 imu_tip = dvec2(true_vehicle.x + g_fusion_scene->imu_measurement.accel.x * 24.0, true_vehicle.y + g_fusion_scene->imu_measurement.accel.y * 24.0);
    DVec2 camera_point = dvec2(gps_corrected.x + matrix_get(&g_fusion_scene->camera_measurement.pose_delta, 0, 0),
                               gps_corrected.y + matrix_get(&g_fusion_scene->camera_measurement.pose_delta, 1, 0));
    MatrixInspector matrix_inspector = {0};
    int linked_row = -1;
    int linked_col = -1;
    bool hovered_cell = false;
    int hover_row = -1;
    int hover_col = -1;

    blueprint_draw_state_trajectory(engine, g_fusion_scene->true_path, g_fusion_scene->path_count, (Color){90, 110, 138, 200});
    blueprint_draw_state_trajectory(engine, g_fusion_scene->predicted_path, g_fusion_scene->path_count, (Color){120, 196, 255, 180});
    blueprint_draw_state_trajectory(engine, g_fusion_scene->fused_path, g_fusion_scene->path_count, (Color){112, 232, 176, 220});

    blueprint_draw_probability_heatmap(engine, &g_fusion_scene->predicted_state, dvec2(predicted.x - 220.0, predicted.y - 220.0), dvec2(predicted.x + 220.0, predicted.y + 220.0), 24, 24, (Color){104, 186, 255, 255}, "predicted density");
    blueprint_draw_probability_heatmap(engine, &g_fusion_scene->fused_state, dvec2(fused.x - 180.0, fused.y - 180.0), dvec2(fused.x + 180.0, fused.y + 180.0), 24, 24, (Color){108, 228, 176, 255}, "fused density");
    blueprint_draw_uncertainty_propagation(engine, &g_fusion_scene->prior_state, &g_fusion_scene->predicted_state, (Color){88, 142, 220, 255}, (Color){154, 220, 255, 255}, "imu prediction");
    blueprint_draw_gaussian_state(engine, &g_fusion_scene->gps_corrected_state, (Color){248, 188, 116, 255}, "gps corrected");
    blueprint_draw_gaussian_state(engine, &g_fusion_scene->fused_state, (Color){112, 232, 176, 255}, "fused state");
    blueprint_draw_measurement_covariance(engine, gps_meas, &g_fusion_scene->gps_sensor.R, (Color){244, 170, 96, 255}, "gps R");
    draw_true_vehicle_marker(engine, &g_fusion_scene->true_vehicle, (Color){236, 236, 242, 255});
    blueprint_draw_signal_arrow(engine, true_vehicle, imu_tip, 1.8f, (Color){118, 208, 255, 255}, 0.0);
    blueprint_draw_signal_arrow(engine, true_vehicle, gps_meas, 1.9f, (Color){248, 176, 102, 255}, 0.2);
    blueprint_draw_signal_arrow(engine, true_vehicle, camera_point, 1.8f, (Color){196, 132, 255, 255}, 0.4);
    draw_timestamp_label(engine, imu_tip, g_fusion_scene->last_imu_sample_time, (Color){118, 208, 255, 255}, "imu");
    if (g_fusion_scene->last_gps_sample_time >= 0.0) {
        draw_timestamp_label(engine, gps_meas, g_fusion_scene->last_gps_sample_time, (Color){248, 176, 102, 255}, "gps");
    }
    if (g_fusion_scene->last_camera_sample_time >= 0.0) {
        draw_timestamp_label(engine, camera_point, g_fusion_scene->last_camera_sample_time, (Color){196, 132, 255, 255}, "cam");
    }
    blueprint_draw_residual_visual(engine, predicted, gps_meas, &g_fusion_scene->gps_residual, (Color){248, 176, 102, 255}, "gps innovation");
    blueprint_draw_residual_visual(engine, gps_corrected, fused, &g_fusion_scene->camera_residual, (Color){196, 132, 255, 255}, "camera delta");
    blueprint_draw_feature_flow(engine, g_fusion_scene->feature_from, g_fusion_scene->feature_to, g_fusion_scene->feature_count, (Color){188, 132, 255, 180});
    blueprint_draw_pose_graph_edge(engine, gps_corrected, fused, (Color){188, 132, 255, 255}, "camera pose edge");
    draw_velocity_arrow(engine, &g_fusion_scene->predicted_state, (Color){120, 196, 255, 255}, "v^-");
    draw_velocity_arrow(engine, &g_fusion_scene->fused_state, (Color){112, 232, 176, 255}, "v^+");

    DVec2 imu_node_center = dvec2(-220.0, -250.0);
    DVec2 gps_node_center = dvec2(340.0, -250.0);
    DVec2 camera_node_center = dvec2(900.0, -250.0);
    draw_sensor_node(engine, &g_fusion_scene->imu_node, imu_node_center, (Vector2){210.0f, 96.0f}, (Color){118, 208, 255, 255}, 0, "IMU Prediction");
    draw_sensor_node(engine, &g_fusion_scene->gps_node, gps_node_center, (Vector2){210.0f, 96.0f}, (Color){248, 176, 102, 255}, 1, "GPS Update");
    draw_sensor_node(engine, &g_fusion_scene->camera_node, camera_node_center, (Vector2){220.0f, 96.0f}, (Color){196, 132, 255, 255}, 2, "Camera VO");
    maybe_describe_fusion_node(engine, imu_node_center, (Vector2){210.0f, 96.0f}, "IMU Prediction", "Continuous motion-model node. Integrates inertial input to predict the next state and propagate covariance with F P F^T + Q.", &matrix_inspector);
    maybe_describe_fusion_node(engine, gps_node_center, (Vector2){210.0f, 96.0f}, "GPS Update", "Absolute-position measurement update. Compares predicted position against GPS, forms innovation/S/K, and pulls the estimate back toward the measurement.", &matrix_inspector);
    maybe_describe_fusion_node(engine, camera_node_center, (Vector2){220.0f, 96.0f}, "Camera VO", "Relative-motion update. Uses visual odometry pose change to refine the GPS-corrected state and tighten uncertainty along observed motion directions.", &matrix_inspector);

    blueprint_draw_tensor_flow_edge(engine, dvec2(-900.0, -70.0), dvec2(imu_node_center.x - 120.0, imu_node_center.y), (Color){118, 208, 255, 255}, "state", fusion_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, dvec2(imu_node_center.x + 120.0, imu_node_center.y), predicted, (Color){118, 208, 255, 255}, "x^-, P^-", fusion_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, gps_meas, dvec2(gps_node_center.x - 124.0, gps_node_center.y), (Color){248, 176, 102, 255}, "gps z", fusion_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, predicted, dvec2(gps_node_center.x - 124.0, gps_node_center.y + 26.0), (Color){248, 176, 102, 255}, "pred", fusion_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, dvec2(gps_node_center.x + 124.0, gps_node_center.y), gps_corrected, (Color){248, 176, 102, 255}, "gps-corrected", fusion_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, gps_corrected, dvec2(camera_node_center.x - 132.0, camera_node_center.y + 28.0), (Color){196, 132, 255, 255}, "prior", fusion_node_active(engine, 2));
    blueprint_draw_tensor_flow_edge(engine, dvec2(camera_node_center.x + 132.0, camera_node_center.y), fused, (Color){196, 132, 255, 255}, "central estimate", fusion_node_active(engine, 2));

    draw_fusion_matrix_panel(engine, &g_fusion_scene->transition_matrix_view, dvec2(-1280.0, 420.0), g_fusion_scene->cell_size, "F", &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_fusion_matrix_panel(engine, &g_fusion_scene->imu_process_view, dvec2(-760.0, 420.0), g_fusion_scene->cell_size, "Q", &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_sensor_matrix_panel(engine, &g_fusion_scene->gps_sensor.H, dvec2(-220.0, 420.0), g_fusion_scene->cell_size, "GPS sensor", "GPS", false, &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_sensor_matrix_panel(engine, &g_fusion_scene->gps_sensor.R, dvec2(-220.0, 420.0 + g_fusion_scene->gps_sensor.H.rows * g_fusion_scene->cell_size + g_fusion_scene->cell_size * 1.3), g_fusion_scene->cell_size, "R_gps", "GPS", true, &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_sensor_matrix_panel(engine, &g_fusion_scene->camera_sensor.H, dvec2(340.0, 420.0), g_fusion_scene->cell_size, "Camera sensor", "Camera", false, &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_sensor_matrix_panel(engine, &g_fusion_scene->camera_sensor.R, dvec2(340.0, 420.0 + g_fusion_scene->camera_sensor.H.rows * g_fusion_scene->cell_size + g_fusion_scene->cell_size * 1.3), g_fusion_scene->cell_size, "R_cam", "Camera", true, &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_gain_matrix_panel(engine, &g_fusion_scene->gps_internals.K, dvec2(900.0, 420.0), g_fusion_scene->cell_size, "K_gps", "GPS", &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_gain_matrix_panel(engine, &g_fusion_scene->camera_internals.K, dvec2(1320.0, 420.0), g_fusion_scene->cell_size, "K_cam", "Camera", &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_innovation_vector_panel(engine, &g_fusion_scene->gps_internals.innovation, dvec2(900.0, 120.0), g_fusion_scene->cell_size * 0.9f, "gps innovation", (Color){248, 176, 102, 255}, "GPS", &matrix_inspector, linked_row, &hovered_cell, &hover_row);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = 0;
    }
    draw_innovation_vector_panel(engine, &g_fusion_scene->camera_internals.innovation, dvec2(1180.0, 120.0), g_fusion_scene->cell_size * 0.9f, "camera innovation", (Color){196, 132, 255, 255}, "Camera", &matrix_inspector, linked_row, &hovered_cell, &hover_row);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = 0;
    }
    draw_residual_matrix_panel(engine, &g_fusion_scene->gps_residual.S, dvec2(620.0, 120.0), g_fusion_scene->cell_size, "S_gps", "GPS", &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_residual_matrix_panel(engine, &g_fusion_scene->camera_residual.S, dvec2(620.0, 420.0), g_fusion_scene->cell_size, "S_cam", "Camera", &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    draw_interactive_covariance_timeline(engine, g_fusion_scene->covariance_history, g_fusion_scene->covariance_history_count, 5, dvec2(-1320.0, 860.0), 18.0f, &matrix_inspector, linked_row, linked_col, &hovered_cell, &hover_row, &hover_col);
    if (hovered_cell) {
        linked_row = hover_row;
        linked_col = hover_col;
    }
    if (linked_row >= 0 && linked_col >= 0) {
        blueprint_draw_matrix_heatmap(engine, &g_fusion_scene->transition_matrix_view, dvec2(-1280.0, 420.0), g_fusion_scene->cell_size, true, linked_row, linked_col, linked_row, linked_col, "F");
        blueprint_draw_matrix_heatmap(engine, &g_fusion_scene->imu_process_view, dvec2(-760.0, 420.0), g_fusion_scene->cell_size, true, linked_row, linked_col, linked_row, linked_col, "Q");
    }
    blueprint_draw_innovation_statistics(engine, g_fusion_scene->gps_innovation_history, g_fusion_scene->gps_innovation_count, dvec2(180.0, -520.0), (Vector2){420.0f, 180.0f}, (Color){248, 176, 102, 255}, "gps innovation stats");
    blueprint_draw_innovation_statistics(engine, g_fusion_scene->camera_innovation_history, g_fusion_scene->camera_innovation_count, dvec2(660.0, -520.0), (Vector2){420.0f, 180.0f}, (Color){196, 132, 255, 255}, "camera innovation stats");
    blueprint_draw_sensor_timing_lanes(engine, g_fusion_scene->imu_event_times, g_fusion_scene->imu_event_count, g_fusion_scene->camera_event_times, g_fusion_scene->camera_event_count, g_fusion_scene->gps_event_times, g_fusion_scene->gps_event_count, engine->time_seconds, dvec2(-1320.0, -520.0), (Vector2){420.0f, 180.0f}, "sensor timing");
    blueprint_draw_large_factor_graph(engine, &g_fusion_scene->factor_graph, g_fusion_scene->vehicle_tracks, g_fusion_scene->vehicle_track_count, "distributed factor graph");
    blueprint_draw_execution_timeline(engine, &g_fusion_scene->debugger, dvec2(180.0, -300.0), (Vector2){900.0f, 210.0f}, "algorithm execution");
    debugger_draw_overlay();
    draw_matrix_inspector_box(&matrix_inspector);
    draw_causal_chain_overlay(engine, g_fusion_scene);

    const char *lines[] = {
        "n/b step, p run-pause, u/o scrub history; the debugger exposes one atomic Kalman operation at a time",
        "IMU drives continuous propagation: x=f(x,u,dt),  P=F P F^T + Q, with internals exposed per step",
        "GPS and camera updates expose K, S, innovation histories, covariance evolution, asynchronous timing, and a multi-vehicle factor graph"
    };
    blueprint_draw_equation_block(engine, dvec2(-1320.0, -540.0), "multi-sensor fusion pipeline", lines, 3, (Color){236, 242, 249, 255});
    {
        const char *graph_lines[] = {
            "Vehicle states are arranged in fixed left-to-right lanes so graph structure stays readable as factors accumulate",
            "GPS anchors occupy a separate upper band; loop closures and cross-vehicle constraints span between stable rows",
            "Zoom in to inspect covariance panels and matrix-valued factors without losing the global graph topology"
        };
        blueprint_draw_equation_block(engine, dvec2(-2320.0, 1320.0), "large-scale estimation graph", graph_lines, 3, (Color){232, 238, 246, 255});
    }
}

static MathSceneData *create_math_scene(void) {
    static const double a_values[] = {1.2, -0.4, 0.8, 0.3, 1.1, 0.5, -0.6, 0.9, 1.4};
    static const double b_values[] = {0.7, 1.0, -0.2, -1.1, 0.4, 0.9, 0.5, -0.7, 1.3};
    static const double covariance_values[] = {4.0, 1.6, 1.6, 2.4};
    static const double transition_values[] = {1.0, 0.2, 0.0, 0.96};
    static const double state_values[] = {1.2, -0.8};
    MathSceneData *scene = calloc(1, sizeof(*scene));
    if (scene == NULL) return NULL;
    scene->cell_size = 60.0f;
    scene->vector_cell_size = 54.0f;
    scene->a = matrix_from_array(3, 3, a_values);
    scene->b = matrix_from_array(3, 3, b_values);
    scene->c = matrix_create(3, 3);
    scene->a_transpose = matrix_create(3, 3);
    scene->covariance = matrix_from_array(2, 2, covariance_values);
    scene->covariance_inverse = matrix_create(2, 2);
    scene->transition = matrix_from_array(2, 2, transition_values);
    scene->propagated_covariance = matrix_create(2, 2);
    scene->state_vector = vector_from_array(2, state_values);
    scene->transformed_vector = vector_create(2);
    scene->state_matrix = matrix_create(2, 1);
    scene->transformed_state_matrix = matrix_create(2, 1);
    if (scene->a == NULL || scene->b == NULL || scene->c == NULL || scene->a_transpose == NULL || scene->covariance == NULL || scene->covariance_inverse == NULL ||
        scene->transition == NULL || scene->propagated_covariance == NULL || scene->state_vector == NULL || scene->transformed_vector == NULL || scene->state_matrix == NULL ||
        scene->transformed_state_matrix == NULL) {
        destroy_math_scene(scene);
        return NULL;
    }
    for (int i = 0; i < scene->state_vector->size; ++i) matrix_set(scene->state_matrix, i, 0, scene->state_vector->data[i]);
    scene->heatmap_a = (TensorHeatmap){scene->a, (Vector2){-1180.0f, -360.0f}, scene->cell_size};
    scene->heatmap_b = (TensorHeatmap){scene->b, (Vector2){-760.0f, -360.0f}, scene->cell_size};
    scene->heatmap_c = (TensorHeatmap){scene->c, (Vector2){20.0f, -360.0f}, scene->cell_size};
    scene->nodes[0] = (MathNode){"matrix multiply", (Vector2){-320.0f, -170.0f}, scene->a, scene->b, scene->c, compute_matrix_multiply_node, draw_multiply_math_node};
    scene->nodes[1] = (MathNode){"transpose", (Vector2){380.0f, -170.0f}, scene->a, NULL, scene->a_transpose, compute_matrix_transpose_node, draw_transpose_math_node};
    scene->nodes[2] = (MathNode){"inverse", (Vector2){-850.0f, 470.0f}, scene->covariance, NULL, scene->covariance_inverse, compute_matrix_inverse_node, draw_inverse_math_node};
    scene->nodes[3] = (MathNode){"vector transform", (Vector2){560.0f, 470.0f}, scene->transition, scene->state_matrix, scene->transformed_state_matrix, compute_vector_transform_node, draw_vector_transform_math_node};
    scene->nodes[4] = (MathNode){"cov propagation", (Vector2){40.0f, 760.0f}, scene->transition, scene->covariance, scene->propagated_covariance, compute_covariance_propagation_node, draw_covariance_math_node};
    scene->node_count = 5;
    scene->scene_min = dvec2(-1300.0, -620.0);
    scene->scene_max = dvec2(980.0, 980.0);
    init_math_scene_compute(scene);
    return scene;
}

static FusionSceneData *create_fusion_scene(void) {
    FusionSceneData *scene = calloc(1, sizeof(*scene));
    if (scene == NULL) return NULL;
    scene->cell_size = 66.0f;
    factor_graph_init(&scene->factor_graph);
    if (!gaussian_state_init(&scene->state_estimate, 5) ||
        !gaussian_state_init(&scene->prior_state, 5) ||
        !gaussian_state_init(&scene->predicted_state, 5) ||
        !gaussian_state_init(&scene->gps_corrected_state, 5) ||
        !gaussian_state_init(&scene->fused_state, 5) ||
        !sensor_model_init(&scene->gps_sensor, "GPS", 2, 5) ||
        !sensor_model_init(&scene->camera_sensor, "Camera VO", 2, 5) ||
        !residual_init(&scene->gps_residual, 2) ||
        !residual_init(&scene->camera_residual, 2) ||
        !camera_measurement_init(&scene->camera_measurement, 3, 1) ||
        !matrix_init_storage(&scene->gps_gain, 5, 2) ||
        !matrix_init_storage(&scene->camera_gain, 5, 2) ||
        !matrix_init_storage(&scene->process_noise, 5, 5) ||
        !matrix_init_storage(&scene->transition_jacobian, 5, 5) ||
        !matrix_init_storage(&scene->transition_matrix_view, 5, 5) ||
        !matrix_init_storage(&scene->imu_process_view, 5, 5) ||
        !matrix_init_storage(&scene->debug_hp, 2, 5) ||
        !matrix_init_storage(&scene->debug_hpt, 5, 2) ||
        !matrix_init_storage(&scene->debug_predicted_measurement, 2, 1) ||
        !matrix_init_storage(&scene->debug_correction, 5, 1) ||
        !kalman_internals_init(&scene->imu_internals, 5, 5) ||
        !kalman_internals_init(&scene->gps_internals, 5, 2) ||
        !kalman_internals_init(&scene->camera_internals, 5, 2)) {
        destroy_fusion_scene(scene);
        return NULL;
    }
    for (int i = 0; i < FUSION_HISTORY; ++i) {
        if (!matrix_init_storage(&scene->covariance_history[i], 5, 5)) {
            destroy_fusion_scene(scene);
            return NULL;
        }
    }
    for (int i = 0; i < DEBUG_FRAMES; ++i) {
        if (!simulation_frame_init(&scene->frame_history[i], 5, 2)) {
            destroy_fusion_scene(scene);
            return NULL;
        }
    }
    if (!simulation_frame_init(&scene->debug_frame, 5, 2)) {
        destroy_fusion_scene(scene);
        return NULL;
    }
    vector_init_storage(&scene->imu_node.measurement, 5);
    vector_init_storage(&scene->gps_node.measurement, 2);
    vector_init_storage(&scene->camera_node.measurement, 2);
    scene->imu_node.state = &scene->predicted_state;
    scene->imu_node.sensor = NULL;
    scene->imu_node.internals = &scene->imu_internals;
    scene->gps_node.state = &scene->gps_corrected_state;
    scene->gps_node.sensor = &scene->gps_sensor;
    scene->gps_node.internals = &scene->gps_internals;
    scene->camera_node.state = &scene->fused_state;
    scene->camera_node.sensor = &scene->camera_sensor;
    scene->camera_node.internals = &scene->camera_internals;
    scene->vehicle_track_count = FUSION_VEHICLES;
    scene->next_anchor_id = GRAPH_ANCHOR_BASE_ID;
    for (int i = 0; i < FUSION_VEHICLES; ++i) {
        scene->vehicle_last_node_id[i] = -1;
        scene->vehicle_next_sample_time[i] = i * 0.07;
        scene->vehicle_tracks[i].vehicle_id = i;
        scene->vehicle_tracks[i].start_node = -1;
        scene->vehicle_tracks[i].end_node = -1;
    }
    execution_timeline_reset(&scene->debugger);
    execution_timeline_add_step(&scene->debugger, "1. prediction", debugger_step_prediction, debugger_draw_overlay);
    execution_timeline_add_step(&scene->debugger, "2. innovation", debugger_step_innovation, debugger_draw_overlay);
    execution_timeline_add_step(&scene->debugger, "3. S matrix", debugger_step_s_matrix, debugger_draw_overlay);
    execution_timeline_add_step(&scene->debugger, "4. kalman gain", debugger_step_gain, debugger_draw_overlay);
    execution_timeline_add_step(&scene->debugger, "5. state correction", debugger_step_state_correction, debugger_draw_overlay);
    execution_timeline_add_step(&scene->debugger, "6. covariance update", debugger_step_covariance_update, debugger_draw_overlay);
    scene->scene_min = dvec2(-2600.0, -620.0);
    scene->scene_max = dvec2(3200.0, 3720.0);
    return scene;
}

static void destroy_math_scene(MathSceneData *scene) {
    if (scene == NULL) return;
    matrix_destroy(scene->a);
    matrix_destroy(scene->b);
    matrix_destroy(scene->c);
    matrix_destroy(scene->a_transpose);
    matrix_destroy(scene->covariance);
    matrix_destroy(scene->covariance_inverse);
    matrix_destroy(scene->transition);
    matrix_destroy(scene->propagated_covariance);
    matrix_destroy(scene->state_matrix);
    matrix_destroy(scene->transformed_state_matrix);
    vector_destroy(scene->state_vector);
    vector_destroy(scene->transformed_vector);
    free(scene);
}

static void destroy_fusion_scene(FusionSceneData *scene) {
    if (scene == NULL) return;
    factor_graph_free(&scene->factor_graph);
    gaussian_state_free(&scene->state_estimate);
    gaussian_state_free(&scene->prior_state);
    gaussian_state_free(&scene->predicted_state);
    gaussian_state_free(&scene->gps_corrected_state);
    gaussian_state_free(&scene->fused_state);
    sensor_model_free(&scene->gps_sensor);
    sensor_model_free(&scene->camera_sensor);
    residual_free(&scene->gps_residual);
    residual_free(&scene->camera_residual);
    camera_measurement_free(&scene->camera_measurement);
    matrix_free_storage(&scene->gps_gain);
    matrix_free_storage(&scene->camera_gain);
    matrix_free_storage(&scene->process_noise);
    matrix_free_storage(&scene->transition_jacobian);
    matrix_free_storage(&scene->transition_matrix_view);
    matrix_free_storage(&scene->imu_process_view);
    matrix_free_storage(&scene->debug_hp);
    matrix_free_storage(&scene->debug_hpt);
    matrix_free_storage(&scene->debug_predicted_measurement);
    matrix_free_storage(&scene->debug_correction);
    kalman_internals_free(&scene->imu_internals);
    kalman_internals_free(&scene->gps_internals);
    kalman_internals_free(&scene->camera_internals);
    for (int i = 0; i < FUSION_HISTORY; ++i) {
        matrix_free_storage(&scene->covariance_history[i]);
    }
    for (int i = 0; i < DEBUG_FRAMES; ++i) {
        simulation_frame_free(&scene->frame_history[i]);
    }
    simulation_frame_free(&scene->debug_frame);
    vector_free_storage(&scene->imu_node.measurement);
    vector_free_storage(&scene->gps_node.measurement);
    vector_free_storage(&scene->camera_node.measurement);
    free(scene);
}

static void add_node(BlueprintEngine *engine, const char *name, BlueprintLayer layer, int page, void (*draw)(Camera2D cam), void (*draw_minimap)(const BlueprintEngine *engine, Rectangle map_rect, DVec2 world_min, DVec2 world_max), DVec2 bounds_min, DVec2 bounds_max) {
    BlueprintNode node = {0};
    strncpy(node.name, name, sizeof(node.name) - 1);
    node.layer = layer;
    node.page = page;
    node.draw = draw;
    node.draw_minimap = draw_minimap;
    node.bounds_min = bounds_min;
    node.bounds_max = bounds_max;
    node.visible = true;
    blueprint_engine_add_node(engine, &node);
}

void blueprint_init_demo(BlueprintEngine *engine) {
    g_math_scene = create_math_scene();
    g_fusion_scene = create_fusion_scene();
    if (g_math_scene == NULL || g_fusion_scene == NULL) {
        fprintf(stderr, "failed to allocate demo scenes\n");
        exit(1);
    }
    add_node(engine, "math-scene", BLUEPRINT_LAYER_MATH, 0, draw_math_scene_node, draw_math_scene_minimap, g_math_scene->scene_min, g_math_scene->scene_max);
    add_node(engine, "fusion-scene", BLUEPRINT_LAYER_MATH, 1, draw_fusion_scene_node, draw_fusion_scene_minimap, g_fusion_scene->scene_min, g_fusion_scene->scene_max);
}

void blueprint_reset_demo(BlueprintEngine *engine) {
    (void)engine;
    destroy_fusion_scene(g_fusion_scene);
    destroy_math_scene(g_math_scene);
    g_fusion_scene = create_fusion_scene();
    g_math_scene = create_math_scene();
    if (g_math_scene == NULL || g_fusion_scene == NULL) {
        fprintf(stderr, "failed to reset demo scenes\n");
        exit(1);
    }
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1440, 900, "Blueprint Math Canvas");
    SetExitKey(KEY_NULL);
    SetTargetFPS(144);

    BlueprintEngine engine;
    blueprint_engine_init(&engine, GetScreenWidth(), GetScreenHeight());
    blueprint_init_demo(&engine);

    while (!WindowShouldClose() && !engine.quit_requested) {
        blueprint_engine_update(&engine, GetFrameTime());
        blueprint_engine_draw(&engine);
    }

    destroy_fusion_scene(g_fusion_scene);
    destroy_math_scene(g_math_scene);
    g_fusion_scene = NULL;
    g_math_scene = NULL;
    blueprint_engine_shutdown(&engine);
    CloseWindow();
    return 0;
}
