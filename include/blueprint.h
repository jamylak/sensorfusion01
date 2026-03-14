#ifndef BLUEPRINT_H
#define BLUEPRINT_H

#include <stdbool.h>
#include <stddef.h>

#define Matrix RaylibMatrix
#include "raylib.h"
#undef Matrix

typedef struct {
    double x;
    double y;
} DVec2;

typedef struct Matrix {
    int rows;
    int cols;
    double *data;
} Matrix;

typedef struct Vector {
    int size;
    double *data;
} Vector;

typedef struct TensorHeatmap {
    Matrix *matrix;
    Vector2 world_position;
    float cell_size;
} TensorHeatmap;

typedef struct GaussianState {
    Vector mean;
    Matrix covariance;
} GaussianState;

typedef struct Measurement {
    Vector z;
    Matrix H;
    Matrix R;
} Measurement;

typedef struct Residual {
    Vector innovation;
    Matrix S;
} Residual;

typedef struct ImuMeasurement {
    Vector3 accel;
    Vector3 gyro;
} ImuMeasurement;

typedef struct GpsMeasurement {
    double lat;
    double lon;
} GpsMeasurement;

typedef struct CameraMeasurement {
    Matrix pose_delta;
} CameraMeasurement;

typedef struct SensorModel {
    char name[64];
    int measurement_dim;
    Matrix H;
    Matrix R;
    Vector (*observe)(Vector state);
} SensorModel;

typedef struct KalmanInternals {
    Matrix F;
    Matrix Q;
    Matrix H;
    Matrix R;
    Matrix K;
    Matrix S;
    Matrix P;
    Vector innovation;
} KalmanInternals;

#define MAX_FACTOR_NODES 2048
#define MAX_FACTOR_EDGES 8192
#define MAX_VEHICLE_TRACKS 8
#define MAX_STEPS 64

typedef struct FactorNode {
    int id;
    Vector state;
    Matrix covariance;
    Vector2 world_position;
} FactorNode;

typedef struct FactorEdge {
    int from_node;
    int to_node;
    Matrix measurement_model;
    Matrix information_matrix;
    Vector residual;
} FactorEdge;

typedef struct FactorGraph {
    FactorNode nodes[MAX_FACTOR_NODES];
    FactorEdge edges[MAX_FACTOR_EDGES];
    int node_count;
    int edge_count;
} FactorGraph;

typedef struct VehicleTrack {
    int vehicle_id;
    int start_node;
    int end_node;
} VehicleTrack;

typedef struct AlgorithmStep {
    char description[128];
    void (*execute)(void);
    void (*draw_overlay)(void);
} AlgorithmStep;

typedef struct ExecutionTimeline {
    AlgorithmStep steps[MAX_STEPS];
    int step_count;
    int current_step;
    bool running;
} ExecutionTimeline;

typedef struct SimulationFrame {
    GaussianState state;
    KalmanInternals internals;
} SimulationFrame;

struct MathNode;
struct EstimationNode;
struct SensorFusionNode;

typedef struct MathNode {
    char name[64];
    Vector2 position;
    Matrix *inputA;
    Matrix *inputB;
    Matrix *output;
    void (*compute)(struct MathNode *);
    void (*draw)(struct MathNode *);
} MathNode;

typedef struct EstimationNode {
    char name[64];
    Vector2 world_position;
    GaussianState *input_state;
    Measurement *measurement;
    GaussianState *output_state;
    void (*compute)(struct EstimationNode *);
    void (*draw)(struct EstimationNode *);
} EstimationNode;

typedef struct SensorFusionNode {
    GaussianState *state;
    SensorModel *sensor;
    Vector measurement;
    KalmanInternals *internals;
    void (*predict)(struct SensorFusionNode *);
    void (*update)(struct SensorFusionNode *);
} SensorFusionNode;

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
    DVec2 bounds_min;
    DVec2 bounds_max;
    void *user_data;
    BlueprintLayer layer;
    int page;
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
    bool quit_requested;
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
bool blueprint_world_rect_visible(const BlueprintEngine *engine, DVec2 min, DVec2 max, double padding);
bool blueprint_world_segment_visible(const BlueprintEngine *engine, DVec2 a, DVec2 b, double padding);
bool blueprint_world_point_visible(const BlueprintEngine *engine, DVec2 point, double padding);

Matrix *matrix_create(int rows, int cols);
Matrix *matrix_from_array(int rows, int cols, const double *values);
void matrix_destroy(Matrix *matrix);
double matrix_get(const Matrix *matrix, int row, int col);
void matrix_set(Matrix *matrix, int row, int col, double value);
bool matrix_multiply_into(const Matrix *a, const Matrix *b, Matrix *out);
bool matrix_transpose_into(const Matrix *in, Matrix *out);
bool matrix_inverse_2x2_into(const Matrix *in, Matrix *out);
bool matrix_covariance_propagate_into(const Matrix *f, const Matrix *cov, Matrix *out);
bool matrix_eigen_2x2(const Matrix *in, double *lambda1, double *lambda2, Vector2 *eigenvector1, Vector2 *eigenvector2);

Vector *vector_create(int size);
Vector *vector_from_array(int size, const double *values);
void vector_destroy(Vector *vector);
bool matrix_transform_vector_into(const Matrix *matrix, const Vector *vector, Vector *out);
bool matrix_init_storage(Matrix *matrix, int rows, int cols);
void matrix_free_storage(Matrix *matrix);
bool vector_init_storage(Vector *vector, int size);
void vector_free_storage(Vector *vector);

bool gaussian_state_init(GaussianState *state, int state_size);
void gaussian_state_free(GaussianState *state);
bool measurement_init(Measurement *measurement, int z_size, int state_size);
void measurement_free(Measurement *measurement);
bool residual_init(Residual *residual, int innovation_size);
void residual_free(Residual *residual);
bool sensor_model_init(SensorModel *sensor, const char *name, int measurement_dim, int state_dim);
void sensor_model_free(SensorModel *sensor);
bool camera_measurement_init(CameraMeasurement *measurement, int rows, int cols);
void camera_measurement_free(CameraMeasurement *measurement);
bool kalman_internals_init(KalmanInternals *internals, int state_dim, int measurement_dim);
void kalman_internals_free(KalmanInternals *internals);
bool factor_graph_init(FactorGraph *graph);
void factor_graph_free(FactorGraph *graph);
FactorNode *factor_graph_add_node(FactorGraph *graph, int id, const Vector *state, const Matrix *covariance, Vector2 world_position);
FactorEdge *factor_graph_add_edge(FactorGraph *graph, int from_node, int to_node, const Matrix *measurement_model, const Matrix *information_matrix, const Vector *residual);
bool factor_graph_add_loop_closure(FactorGraph *graph, int from_node, int to_node, const Matrix *measurement_model, const Matrix *information_matrix, const Vector *residual);
bool simulation_frame_init(SimulationFrame *frame, int state_dim, int measurement_dim);
void simulation_frame_free(SimulationFrame *frame);
void execution_timeline_reset(ExecutionTimeline *timeline);
bool execution_timeline_add_step(ExecutionTimeline *timeline, const char *description, void (*execute)(void), void (*draw_overlay)(void));
void execution_timeline_step_forward(ExecutionTimeline *timeline);
void execution_timeline_step_backward(ExecutionTimeline *timeline);

void blueprint_draw_world_grid(const BlueprintEngine *engine);
void blueprint_draw_arrow(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color);
void blueprint_draw_signal_arrow(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color, double phase_offset);
void blueprint_draw_directed_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color, bool curved);
void blueprint_draw_matrix_grid(const BlueprintEngine *engine, DVec2 origin, int rows, int cols, double cell_size, Color line_color, float thickness);
void blueprint_draw_matrix_heatmap(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, bool show_values, int highlight_row, int highlight_col, int focus_row, int focus_col, const char *title);
void blueprint_draw_vector_visual(const BlueprintEngine *engine, const Vector *vector, DVec2 origin, float cell_size, bool show_values, const char *title, Color accent);
void blueprint_draw_tensor_heatmap(const BlueprintEngine *engine, const TensorHeatmap *heatmap, bool show_values, const char *title);
void blueprint_draw_math_node_box(const BlueprintEngine *engine, const MathNode *node, Vector2 size, Color accent, bool active);
void blueprint_draw_tensor_flow_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, Color color, const char *label, bool active);
void blueprint_draw_matrix_multiply_visualizer(const BlueprintEngine *engine, const Matrix *a, const Matrix *b, const Matrix *c, DVec2 origin_a, DVec2 origin_b, DVec2 origin_c, float cell_size, double time_seconds);
void blueprint_draw_covariance_matrix_visual(const BlueprintEngine *engine, const Matrix *covariance, DVec2 matrix_origin, DVec2 ellipse_origin, float cell_size, const char *title);
void blueprint_draw_gaussian_state(const BlueprintEngine *engine, const GaussianState *state, Color color, const char *label);
void blueprint_draw_probability_heatmap(const BlueprintEngine *engine, const GaussianState *state, DVec2 min, DVec2 max, int steps_x, int steps_y, Color tint, const char *label);
void blueprint_draw_residual_visual(const BlueprintEngine *engine, DVec2 predicted, DVec2 measured, const Residual *residual, Color color, const char *label);
void blueprint_draw_uncertainty_propagation(const BlueprintEngine *engine, const GaussianState *prior, const GaussianState *propagated, Color prior_color, Color propagated_color, const char *label);
void blueprint_draw_estimation_node_box(const BlueprintEngine *engine, const EstimationNode *node, Vector2 size, Color accent, bool active);
void blueprint_draw_sensor_model_box(const BlueprintEngine *engine, const SensorModel *sensor, DVec2 origin, float cell_size, const char *title);
void blueprint_draw_sensor_fusion_node_box(const BlueprintEngine *engine, const SensorFusionNode *node, DVec2 center, Vector2 size, Color accent, bool active, const char *title);
void blueprint_draw_measurement_covariance(const BlueprintEngine *engine, DVec2 center, const Matrix *covariance, Color color, const char *label);
void blueprint_draw_state_trajectory(const BlueprintEngine *engine, const DVec2 *points, int count, Color color);
void blueprint_draw_feature_flow(const BlueprintEngine *engine, const DVec2 *from_points, const DVec2 *to_points, int count, Color color);
void blueprint_draw_pose_graph_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, Color color, const char *label);
void blueprint_draw_kalman_gain_heatmap(const BlueprintEngine *engine, const KalmanInternals *internals, DVec2 origin, float cell_size, const char *title);
void blueprint_draw_covariance_timeline(const BlueprintEngine *engine, const Matrix *history, int history_count, int matrix_dim, DVec2 origin, float cell_size, const char *title);
void blueprint_draw_innovation_statistics(const BlueprintEngine *engine, const DVec2 *samples, int sample_count, DVec2 origin, Vector2 size, Color color, const char *title);
void blueprint_draw_sensor_timing_lanes(const BlueprintEngine *engine, const double *imu_times, int imu_count, const double *camera_times, int camera_count, const double *gps_times, int gps_count, double current_time, DVec2 origin, Vector2 size, const char *title);
void blueprint_draw_factor_graph(const BlueprintEngine *engine, const DVec2 *states, int state_count, const DVec2 *gps_points, const double *gps_residuals, int gps_count, const DVec2 *camera_from, const DVec2 *camera_to, const double *camera_weights, int camera_count, const char *title);
void blueprint_draw_large_factor_graph(const BlueprintEngine *engine, const FactorGraph *graph, const VehicleTrack *tracks, int track_count, const char *title);
void blueprint_draw_execution_timeline(const BlueprintEngine *engine, const ExecutionTimeline *timeline, DVec2 origin, Vector2 size, const char *title);
void blueprint_draw_debug_inspector(const BlueprintEngine *engine, const char *step_description, const Matrix *a, const Matrix *b, const Matrix *c, const Vector *vector, const SensorModel *sensor, DVec2 origin, float cell_size);
bool estimation_prediction_step(const Matrix *F, const GaussianState *input, GaussianState *output);
bool estimation_innovation_step(const GaussianState *predicted, const Measurement *measurement, Residual *residual);
bool estimation_kalman_gain_step(const GaussianState *predicted, const Measurement *measurement, const Residual *residual, Matrix *kalman_gain);
bool estimation_measurement_update_step(const GaussianState *predicted, const Residual *residual, const Measurement *measurement, const Matrix *kalman_gain, GaussianState *corrected);
bool imu_propagation_step(const GaussianState *input, const ImuMeasurement *imu, double dt, const Matrix *process_noise, GaussianState *output, Matrix *out_f);
bool gps_measurement_step(const GaussianState *predicted, const SensorModel *gps_sensor, const GpsMeasurement *gps, Residual *residual, Matrix *kalman_gain, GaussianState *corrected);
bool camera_measurement_step(const GaussianState *predicted, const SensorModel *camera_sensor, const CameraMeasurement *camera, Residual *residual, Matrix *kalman_gain, GaussianState *corrected);
void blueprint_draw_heatmap(const BlueprintEngine *engine, DVec2 origin, const HeatmapData *heatmap);
void blueprint_draw_equation_block(const BlueprintEngine *engine, DVec2 origin, const char *title, const char *lines[], int line_count, Color color);
void blueprint_draw_dense_cluster(const BlueprintEngine *engine, const GraphClusterData *cluster, DVec2 origin, Color node_color, Color edge_color);
void blueprint_draw_covariance_ellipse(const BlueprintEngine *engine, DVec2 origin, const CovarianceData *cov, Color color);

#endif
