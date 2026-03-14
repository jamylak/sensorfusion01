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

static MathSceneData *g_math_scene = NULL;
static FusionSceneData *g_fusion_scene = NULL;

static void destroy_math_scene(MathSceneData *scene);
static void destroy_fusion_scene(FusionSceneData *scene);
static void append_factor_graph_samples(FusionSceneData *scene, double t);
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
    blueprint_draw_matrix_multiply_visualizer(engine, g_math_scene->a, g_math_scene->b, g_math_scene->c,
                                              dvec2(g_math_scene->heatmap_a.world_position.x, g_math_scene->heatmap_a.world_position.y),
                                              dvec2(g_math_scene->heatmap_b.world_position.x, g_math_scene->heatmap_b.world_position.y),
                                              dvec2(g_math_scene->heatmap_c.world_position.x, g_math_scene->heatmap_c.world_position.y),
                                              g_math_scene->cell_size, engine->time_seconds);
    blueprint_draw_matrix_heatmap(engine, g_math_scene->a_transpose, dvec2(560.0, -360.0), g_math_scene->cell_size, true, -1, -1, -1, -1, "A^T");
    blueprint_draw_covariance_matrix_visual(engine, g_math_scene->covariance, dvec2(-1120.0, 300.0), dvec2(-900.0, 460.0), g_math_scene->cell_size, "P");
    blueprint_draw_matrix_heatmap(engine, g_math_scene->covariance_inverse, dvec2(-650.0, 300.0), g_math_scene->cell_size, true, -1, -1, -1, -1, "P^-1");
    blueprint_draw_matrix_heatmap(engine, g_math_scene->transition, dvec2(-80.0, 300.0), g_math_scene->cell_size, true, -1, -1, -1, -1, "F");
    blueprint_draw_vector_visual(engine, g_math_scene->state_vector, dvec2(300.0, 270.0), g_math_scene->vector_cell_size, true, "x", (Color){124, 228, 196, 255});
    blueprint_draw_vector_visual(engine, g_math_scene->transformed_vector, dvec2(780.0, 270.0), g_math_scene->vector_cell_size, true, "Fx", (Color){246, 168, 96, 255});
    blueprint_draw_covariance_matrix_visual(engine, g_math_scene->propagated_covariance, dvec2(360.0, 640.0), dvec2(590.0, 800.0), g_math_scene->cell_size, "P'");
    for (int i = 0; i < g_math_scene->node_count; ++i) g_math_scene->nodes[i].draw(&g_math_scene->nodes[i]);
    draw_math_graph(engine, g_math_scene);
    const char *lines[] = {
        "Explicit matrix/tensor graph for estimation and control pipelines",
        "Every matrix is rendered as a full grid; no icon substitution",
        "MathNode chain: multiply, transpose, inverse, vector transform, covariance propagation"
    };
    blueprint_draw_equation_block(engine, dvec2(-1220.0, -560.0), "linear algebra layer", lines, 3, (Color){235, 242, 249, 255});
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
    scene->last_true_position = dvec2(scene->state_estimate.mean.data[0], scene->state_estimate.mean.data[1]);
    scene->last_theta = scene->state_estimate.mean.data[4];
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

    if (scene->camera_measurement.pose_delta.rows >= 2) {
        matrix_set(&scene->camera_measurement.pose_delta, 0, 0, (true_x - scene->last_true_position.x) + 6.0 * sin(t * 0.8));
        matrix_set(&scene->camera_measurement.pose_delta, 1, 0, (true_y - scene->last_true_position.y) + 6.0 * cos(t * 0.7));
        if (scene->camera_measurement.pose_delta.rows > 2) {
            matrix_set(&scene->camera_measurement.pose_delta, 2, 0, true_theta - scene->state_estimate.mean.data[4]);
        }
    }
    scene->last_true_position = dvec2(true_x, true_y);
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
    copy_gaussian_state(&scene->state_estimate, &scene->fused_state);
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

    blueprint_draw_state_trajectory(engine, g_fusion_scene->true_path, g_fusion_scene->path_count, (Color){90, 110, 138, 200});
    blueprint_draw_state_trajectory(engine, g_fusion_scene->predicted_path, g_fusion_scene->path_count, (Color){120, 196, 255, 180});
    blueprint_draw_state_trajectory(engine, g_fusion_scene->fused_path, g_fusion_scene->path_count, (Color){112, 232, 176, 220});

    blueprint_draw_probability_heatmap(engine, &g_fusion_scene->predicted_state, dvec2(predicted.x - 220.0, predicted.y - 220.0), dvec2(predicted.x + 220.0, predicted.y + 220.0), 24, 24, (Color){104, 186, 255, 255}, "predicted density");
    blueprint_draw_probability_heatmap(engine, &g_fusion_scene->fused_state, dvec2(fused.x - 180.0, fused.y - 180.0), dvec2(fused.x + 180.0, fused.y + 180.0), 24, 24, (Color){108, 228, 176, 255}, "fused density");
    blueprint_draw_uncertainty_propagation(engine, &g_fusion_scene->prior_state, &g_fusion_scene->predicted_state, (Color){88, 142, 220, 255}, (Color){154, 220, 255, 255}, "imu prediction");
    blueprint_draw_gaussian_state(engine, &g_fusion_scene->gps_corrected_state, (Color){248, 188, 116, 255}, "gps corrected");
    blueprint_draw_gaussian_state(engine, &g_fusion_scene->fused_state, (Color){112, 232, 176, 255}, "fused state");
    blueprint_draw_measurement_covariance(engine, gps_meas, &g_fusion_scene->gps_sensor.R, (Color){244, 170, 96, 255}, "gps R");
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

    blueprint_draw_tensor_flow_edge(engine, dvec2(-900.0, -70.0), dvec2(imu_node_center.x - 120.0, imu_node_center.y), (Color){118, 208, 255, 255}, "state", fusion_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, dvec2(imu_node_center.x + 120.0, imu_node_center.y), predicted, (Color){118, 208, 255, 255}, "x^-, P^-", fusion_node_active(engine, 0));
    blueprint_draw_tensor_flow_edge(engine, gps_meas, dvec2(gps_node_center.x - 124.0, gps_node_center.y), (Color){248, 176, 102, 255}, "gps z", fusion_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, predicted, dvec2(gps_node_center.x - 124.0, gps_node_center.y + 26.0), (Color){248, 176, 102, 255}, "pred", fusion_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, dvec2(gps_node_center.x + 124.0, gps_node_center.y), gps_corrected, (Color){248, 176, 102, 255}, "gps-corrected", fusion_node_active(engine, 1));
    blueprint_draw_tensor_flow_edge(engine, gps_corrected, dvec2(camera_node_center.x - 132.0, camera_node_center.y + 28.0), (Color){196, 132, 255, 255}, "prior", fusion_node_active(engine, 2));
    blueprint_draw_tensor_flow_edge(engine, dvec2(camera_node_center.x + 132.0, camera_node_center.y), fused, (Color){196, 132, 255, 255}, "central estimate", fusion_node_active(engine, 2));

    blueprint_draw_matrix_heatmap(engine, &g_fusion_scene->transition_matrix_view, dvec2(-1280.0, 420.0), g_fusion_scene->cell_size, true, -1, -1, -1, -1, "F");
    blueprint_draw_matrix_heatmap(engine, &g_fusion_scene->imu_process_view, dvec2(-760.0, 420.0), g_fusion_scene->cell_size, true, -1, -1, -1, -1, "Q");
    blueprint_draw_sensor_model_box(engine, &g_fusion_scene->gps_sensor, dvec2(-220.0, 420.0), g_fusion_scene->cell_size, "GPS sensor");
    blueprint_draw_sensor_model_box(engine, &g_fusion_scene->camera_sensor, dvec2(340.0, 420.0), g_fusion_scene->cell_size, "Camera sensor");
    blueprint_draw_kalman_gain_heatmap(engine, &g_fusion_scene->gps_internals, dvec2(900.0, 420.0), g_fusion_scene->cell_size, "K_gps");
    blueprint_draw_kalman_gain_heatmap(engine, &g_fusion_scene->camera_internals, dvec2(1320.0, 420.0), g_fusion_scene->cell_size, "K_cam");
    blueprint_draw_vector_visual(engine, &g_fusion_scene->gps_internals.innovation, dvec2(900.0, 120.0), g_fusion_scene->cell_size * 0.9f, true, "gps innovation", (Color){248, 176, 102, 255});
    blueprint_draw_vector_visual(engine, &g_fusion_scene->camera_internals.innovation, dvec2(1180.0, 120.0), g_fusion_scene->cell_size * 0.9f, true, "camera innovation", (Color){196, 132, 255, 255});
    blueprint_draw_matrix_heatmap(engine, &g_fusion_scene->gps_residual.S, dvec2(620.0, 120.0), g_fusion_scene->cell_size, true, -1, -1, -1, -1, "S_gps");
    blueprint_draw_matrix_heatmap(engine, &g_fusion_scene->camera_residual.S, dvec2(620.0, 420.0), g_fusion_scene->cell_size, true, -1, -1, -1, -1, "S_cam");
    blueprint_draw_covariance_timeline(engine, g_fusion_scene->covariance_history, g_fusion_scene->covariance_history_count, 5, dvec2(-1320.0, 760.0), 18.0f, "covariance evolution");
    blueprint_draw_innovation_statistics(engine, g_fusion_scene->gps_innovation_history, g_fusion_scene->gps_innovation_count, dvec2(180.0, -520.0), (Vector2){420.0f, 180.0f}, (Color){248, 176, 102, 255}, "gps innovation stats");
    blueprint_draw_innovation_statistics(engine, g_fusion_scene->camera_innovation_history, g_fusion_scene->camera_innovation_count, dvec2(660.0, -520.0), (Vector2){420.0f, 180.0f}, (Color){196, 132, 255, 255}, "camera innovation stats");
    blueprint_draw_sensor_timing_lanes(engine, g_fusion_scene->imu_event_times, g_fusion_scene->imu_event_count, g_fusion_scene->camera_event_times, g_fusion_scene->camera_event_count, g_fusion_scene->gps_event_times, g_fusion_scene->gps_event_count, engine->time_seconds, dvec2(-1320.0, -520.0), (Vector2){420.0f, 180.0f}, "sensor timing");
    blueprint_draw_large_factor_graph(engine, &g_fusion_scene->factor_graph, g_fusion_scene->vehicle_tracks, g_fusion_scene->vehicle_track_count, "distributed factor graph");
    blueprint_draw_execution_timeline(engine, &g_fusion_scene->debugger, dvec2(180.0, -300.0), (Vector2){900.0f, 210.0f}, "algorithm execution");
    debugger_draw_overlay();

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

static void add_node(BlueprintEngine *engine, const char *name, BlueprintLayer layer, int page, void (*draw)(Camera2D cam), DVec2 bounds_min, DVec2 bounds_max) {
    BlueprintNode node = {0};
    strncpy(node.name, name, sizeof(node.name) - 1);
    node.layer = layer;
    node.page = page;
    node.draw = draw;
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
    add_node(engine, "math-scene", BLUEPRINT_LAYER_MATH, 0, draw_math_scene_node, g_math_scene->scene_min, g_math_scene->scene_max);
    add_node(engine, "fusion-scene", BLUEPRINT_LAYER_MATH, 1, draw_fusion_scene_node, g_fusion_scene->scene_min, g_fusion_scene->scene_max);
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
