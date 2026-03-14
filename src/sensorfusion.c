#include "blueprint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DVec2 dvec2(double x, double y) {
    DVec2 v = {x, y};
    return v;
}

static Color color_lerp(Color a, Color b, float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    Color out = {
        (unsigned char)(a.r + (b.r - a.r) * t),
        (unsigned char)(a.g + (b.g - a.g) * t),
        (unsigned char)(a.b + (b.b - a.b) * t),
        (unsigned char)(a.a + (b.a - a.a) * t)
    };
    return out;
}

static bool matrix_add_scaled_diag(Matrix *matrix, const Matrix *diag) {
    if (matrix == NULL || diag == NULL || matrix->rows != diag->rows || matrix->cols != diag->cols) {
        return false;
    }
    for (int i = 0; i < matrix->rows * matrix->cols; ++i) {
        matrix->data[i] += diag->data[i];
    }
    return true;
}

static void matrix_set_identity(Matrix *matrix) {
    memset(matrix->data, 0, sizeof(*matrix->data) * (size_t)matrix->rows * (size_t)matrix->cols);
    for (int i = 0; i < matrix->rows && i < matrix->cols; ++i) {
        matrix_set(matrix, i, i, 1.0);
    }
}

bool sensor_model_init(SensorModel *sensor, const char *name, int measurement_dim, int state_dim) {
    if (sensor == NULL) {
        return false;
    }
    memset(sensor, 0, sizeof(*sensor));
    if (name != NULL) {
        strncpy(sensor->name, name, sizeof(sensor->name) - 1);
    }
    sensor->measurement_dim = measurement_dim;
    if (!matrix_init_storage(&sensor->H, measurement_dim, state_dim)) {
        return false;
    }
    if (!matrix_init_storage(&sensor->R, measurement_dim, measurement_dim)) {
        matrix_free_storage(&sensor->H);
        return false;
    }
    return true;
}

void sensor_model_free(SensorModel *sensor) {
    if (sensor == NULL) {
        return;
    }
    matrix_free_storage(&sensor->H);
    matrix_free_storage(&sensor->R);
}

bool camera_measurement_init(CameraMeasurement *measurement, int rows, int cols) {
    if (measurement == NULL) {
        return false;
    }
    memset(measurement, 0, sizeof(*measurement));
    return matrix_init_storage(&measurement->pose_delta, rows, cols);
}

void camera_measurement_free(CameraMeasurement *measurement) {
    if (measurement == NULL) {
        return;
    }
    matrix_free_storage(&measurement->pose_delta);
}

bool kalman_internals_init(KalmanInternals *internals, int state_dim, int measurement_dim) {
    if (internals == NULL) {
        return false;
    }
    memset(internals, 0, sizeof(*internals));
    return matrix_init_storage(&internals->F, state_dim, state_dim) &&
           matrix_init_storage(&internals->Q, state_dim, state_dim) &&
           matrix_init_storage(&internals->H, measurement_dim, state_dim) &&
           matrix_init_storage(&internals->R, measurement_dim, measurement_dim) &&
           matrix_init_storage(&internals->K, state_dim, measurement_dim) &&
           matrix_init_storage(&internals->S, measurement_dim, measurement_dim) &&
           matrix_init_storage(&internals->P, state_dim, state_dim) &&
           vector_init_storage(&internals->innovation, measurement_dim);
}

void kalman_internals_free(KalmanInternals *internals) {
    if (internals == NULL) {
        return;
    }
    matrix_free_storage(&internals->F);
    matrix_free_storage(&internals->Q);
    matrix_free_storage(&internals->H);
    matrix_free_storage(&internals->R);
    matrix_free_storage(&internals->K);
    matrix_free_storage(&internals->S);
    matrix_free_storage(&internals->P);
    vector_free_storage(&internals->innovation);
}

void blueprint_draw_sensor_model_box(const BlueprintEngine *engine, const SensorModel *sensor, DVec2 origin, float cell_size, const char *title) {
    blueprint_draw_matrix_heatmap(engine, &sensor->H, origin, cell_size, true, -1, -1, -1, -1, title != NULL ? title : sensor->name);
    blueprint_draw_matrix_heatmap(engine, &sensor->R, dvec2(origin.x, origin.y + sensor->H.rows * cell_size + cell_size * 1.3), cell_size, true, -1, -1, -1, -1, "R");
}

void blueprint_draw_sensor_fusion_node_box(const BlueprintEngine *engine, const SensorFusionNode *node, DVec2 center, Vector2 size, Color accent, bool active, const char *title) {
    DVec2 origin = dvec2(center.x - size.x * 0.5, center.y - size.y * 0.5);
    Vector2 top_left = blueprint_world_to_screen(engine, origin);
    Rectangle rect = {top_left.x, top_left.y, blueprint_world_length_to_screen(engine, size.x), blueprint_world_length_to_screen(engine, size.y)};
    DrawRectangleRounded(rect, 0.12f, 8, active ? color_lerp(accent, WHITE, 0.18f) : Fade((Color){20, 26, 34, 255}, 0.95f));
    DrawRectangleRoundedLinesEx(rect, 0.12f, 8, active ? 2.4f : 1.4f, accent);
    DrawText(title != NULL ? title : (node->sensor != NULL ? node->sensor->name : "fusion"), (int)rect.x + 10, (int)rect.y + 10, 15, (Color){232, 238, 246, 255});
    if (node->sensor != NULL) {
        DrawText(node->sensor->name, (int)rect.x + 10, (int)rect.y + 32, 13, color_lerp(accent, WHITE, 0.2f));
    }
}

void blueprint_draw_measurement_covariance(const BlueprintEngine *engine, DVec2 center, const Matrix *covariance, Color color, const char *label) {
    GaussianState temp = {0};
    temp.mean.size = 2;
    double mean_data[2] = {center.x, center.y};
    temp.mean.data = mean_data;
    temp.covariance = *covariance;
    blueprint_draw_gaussian_state(engine, &temp, color, label);
}

void blueprint_draw_state_trajectory(const BlueprintEngine *engine, const DVec2 *points, int count, Color color) {
    for (int i = 1; i < count; ++i) {
        if (!blueprint_world_segment_visible(engine, points[i - 1], points[i], 8.0)) {
            continue;
        }
        Vector2 a = blueprint_world_to_screen(engine, points[i - 1]);
        Vector2 b = blueprint_world_to_screen(engine, points[i]);
        DrawLineEx(a, b, 2.0f, color);
    }
}

void blueprint_draw_feature_flow(const BlueprintEngine *engine, const DVec2 *from_points, const DVec2 *to_points, int count, Color color) {
    for (int i = 0; i < count; ++i) {
        if (!blueprint_world_segment_visible(engine, from_points[i], to_points[i], 4.0)) {
            continue;
        }
        blueprint_draw_arrow(engine, from_points[i], to_points[i], 1.1f, color);
    }
}

void blueprint_draw_pose_graph_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, Color color, const char *label) {
    blueprint_draw_directed_edge(engine, from, to, 1.8f, color, true);
    if (label != NULL) {
        DVec2 mid = dvec2((from.x + to.x) * 0.5, (from.y + to.y) * 0.5);
        Vector2 p = blueprint_world_to_screen(engine, mid);
        DrawText(label, (int)p.x + 8, (int)p.y - 8, 13, color_lerp(color, WHITE, 0.25f));
    }
}

void blueprint_draw_kalman_gain_heatmap(const BlueprintEngine *engine, const KalmanInternals *internals, DVec2 origin, float cell_size, const char *title) {
    blueprint_draw_matrix_heatmap(engine, &internals->K, origin, cell_size, true, -1, -1, -1, -1, title != NULL ? title : "K");
}

void blueprint_draw_covariance_timeline(const BlueprintEngine *engine, const Matrix *history, int history_count, int matrix_dim, DVec2 origin, float cell_size, const char *title) {
    if (title != NULL) {
        Vector2 p = blueprint_world_to_screen(engine, dvec2(origin.x, origin.y - cell_size * 0.9));
        DrawText(title, (int)p.x, (int)p.y, 15, (Color){232, 238, 246, 255});
    }
    for (int i = 0; i < history_count; ++i) {
        char label[32];
        snprintf(label, sizeof(label), "P[%d]", i);
        blueprint_draw_matrix_heatmap(engine, &history[i], dvec2(origin.x + i * (matrix_dim * cell_size + cell_size * 0.8), origin.y), cell_size, false, -1, -1, -1, -1, label);
    }
}

void blueprint_draw_innovation_statistics(const BlueprintEngine *engine, const DVec2 *samples, int sample_count, DVec2 origin, Vector2 size, Color color, const char *title) {
    Vector2 a = blueprint_world_to_screen(engine, origin);
    Vector2 b = blueprint_world_to_screen(engine, dvec2(origin.x + size.x, origin.y + size.y));
    Rectangle rect = {a.x, a.y, b.x - a.x, b.y - a.y};
    DrawRectangleLinesEx(rect, 1.2f, color_lerp(color, WHITE, 0.1f));
    if (title != NULL) {
        DrawText(title, (int)rect.x + 6, (int)rect.y + 6, 14, color_lerp(color, WHITE, 0.15f));
    }
    if (sample_count < 2) {
        return;
    }
    for (int i = 1; i < sample_count; ++i) {
        float x0 = rect.x + ((float)(i - 1) / (float)(sample_count - 1)) * rect.width;
        float x1 = rect.x + ((float)i / (float)(sample_count - 1)) * rect.width;
        float y0 = rect.y + rect.height * 0.5f - (float)(samples[i - 1].y * 0.12);
        float y1 = rect.y + rect.height * 0.5f - (float)(samples[i].y * 0.12);
        DrawLineEx((Vector2){x0, y0}, (Vector2){x1, y1}, 1.6f, color);
        float yy0 = rect.y + rect.height * 0.5f - (float)(samples[i - 1].x * 0.12);
        float yy1 = rect.y + rect.height * 0.5f - (float)(samples[i].x * 0.12);
        DrawLineEx((Vector2){x0, yy0}, (Vector2){x1, yy1}, 1.0f, Fade(color_lerp(color, WHITE, 0.35f), 0.8f));
    }
}

void blueprint_draw_sensor_timing_lanes(const BlueprintEngine *engine, const double *imu_times, int imu_count, const double *camera_times, int camera_count, const double *gps_times, int gps_count, double current_time, DVec2 origin, Vector2 size, const char *title) {
    Vector2 a = blueprint_world_to_screen(engine, origin);
    Vector2 b = blueprint_world_to_screen(engine, dvec2(origin.x + size.x, origin.y + size.y));
    Rectangle rect = {a.x, a.y, b.x - a.x, b.y - a.y};
    DrawRectangleLinesEx(rect, 1.2f, (Color){112, 126, 144, 255});
    if (title != NULL) {
        DrawText(title, (int)rect.x + 6, (int)rect.y + 6, 14, (Color){228, 236, 244, 255});
    }
    const char *labels[3] = {"IMU", "Camera", "GPS"};
    const double *times[3] = {imu_times, camera_times, gps_times};
    const int counts[3] = {imu_count, camera_count, gps_count};
    const Color colors[3] = {
        (Color){110, 202, 255, 255},
        (Color){188, 132, 255, 255},
        (Color){246, 176, 96, 255}
    };
    double window = 4.0;
    for (int lane = 0; lane < 3; ++lane) {
        float y = rect.y + 30.0f + lane * (rect.height - 40.0f) / 3.0f;
        DrawText(labels[lane], (int)rect.x + 8, (int)y - 8, 13, colors[lane]);
        DrawLineEx((Vector2){rect.x + 60.0f, y}, (Vector2){rect.x + rect.width - 8.0f, y}, 1.0f, Fade(colors[lane], 0.4f));
        for (int i = 0; i < counts[lane]; ++i) {
            double age = current_time - times[lane][i];
            if (age < 0.0 || age > window) {
                continue;
            }
            float tx = rect.x + 60.0f + (float)((1.0 - age / window) * (rect.width - 72.0f));
            DrawLineEx((Vector2){tx, y - 8.0f}, (Vector2){tx, y + 8.0f}, 2.0f, colors[lane]);
        }
    }
}

void blueprint_draw_factor_graph(const BlueprintEngine *engine, const DVec2 *states, int state_count, const DVec2 *gps_points, const double *gps_residuals, int gps_count, const DVec2 *camera_from, const DVec2 *camera_to, const double *camera_weights, int camera_count, const char *title) {
    if (title != NULL && state_count > 0) {
        Vector2 p = blueprint_world_to_screen(engine, dvec2(states[0].x, states[0].y - 70.0));
        DrawText(title, (int)p.x, (int)p.y, 14, (Color){232, 238, 246, 255});
    }
    for (int i = 1; i < state_count; ++i) {
        blueprint_draw_pose_graph_edge(engine, states[i - 1], states[i], (Color){100, 150, 210, 180}, "state");
    }
    for (int i = 0; i < gps_count && i < state_count; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "gps r=%.1f", gps_residuals[i]);
        blueprint_draw_pose_graph_edge(engine, states[i], gps_points[i], (Color){246, 176, 96, 220}, label);
    }
    for (int i = 0; i < camera_count; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "w=%.2f", camera_weights[i]);
        blueprint_draw_pose_graph_edge(engine, camera_from[i], camera_to[i], (Color){188, 132, 255, 220}, label);
    }
    for (int i = 0; i < state_count; ++i) {
        Vector2 p = blueprint_world_to_screen(engine, states[i]);
        DrawCircleV(p, 3.5f, (Color){220, 228, 236, 255});
    }
}

bool imu_propagation_step(const GaussianState *input, const ImuMeasurement *imu, double dt, const Matrix *process_noise, GaussianState *output, Matrix *out_f) {
    if (input == NULL || imu == NULL || process_noise == NULL || output == NULL || out_f == NULL || input->mean.size < 5 || output->mean.size < 5 || out_f->rows != 5 || out_f->cols != 5) {
        return false;
    }
    for (int i = 0; i < input->mean.size; ++i) {
        output->mean.data[i] = input->mean.data[i];
    }
    double x = input->mean.data[0];
    double y = input->mean.data[1];
    double vx = input->mean.data[2];
    double vy = input->mean.data[3];
    double theta = input->mean.data[4];
    double ax = imu->accel.x;
    double ay = imu->accel.y;
    double omega = imu->gyro.z;
    double c = cos(theta);
    double s = sin(theta);
    double ax_world = c * ax - s * ay;
    double ay_world = s * ax + c * ay;

    output->mean.data[0] = x + vx * dt + 0.5 * ax_world * dt * dt;
    output->mean.data[1] = y + vy * dt + 0.5 * ay_world * dt * dt;
    output->mean.data[2] = vx + ax_world * dt;
    output->mean.data[3] = vy + ay_world * dt;
    output->mean.data[4] = theta + omega * dt;

    matrix_set_identity(out_f);
    matrix_set(out_f, 0, 2, dt);
    matrix_set(out_f, 1, 3, dt);
    matrix_set(out_f, 0, 4, (-s * ax - c * ay) * 0.5 * dt * dt);
    matrix_set(out_f, 1, 4, ( c * ax - s * ay) * 0.5 * dt * dt);
    matrix_set(out_f, 2, 4, (-s * ax - c * ay) * dt);
    matrix_set(out_f, 3, 4, ( c * ax - s * ay) * dt);

    if (!matrix_covariance_propagate_into(out_f, &input->covariance, &output->covariance)) {
        return false;
    }
    return matrix_add_scaled_diag(&output->covariance, process_noise);
}

static bool measurement_update_internal(const GaussianState *predicted, const SensorModel *sensor, const Vector *measurement_vector, const Matrix *measurement_override_h, Residual *residual, Matrix *kalman_gain, GaussianState *corrected) {
    Measurement m = {0};
    m.z = *measurement_vector;
    m.H = measurement_override_h != NULL ? *measurement_override_h : sensor->H;
    m.R = sensor->R;
    return estimation_innovation_step(predicted, &m, residual) &&
           estimation_kalman_gain_step(predicted, &m, residual, kalman_gain) &&
           estimation_measurement_update_step(predicted, residual, &m, kalman_gain, corrected);
}

bool gps_measurement_step(const GaussianState *predicted, const SensorModel *gps_sensor, const GpsMeasurement *gps, Residual *residual, Matrix *kalman_gain, GaussianState *corrected) {
    Vector z = {0};
    double values[2] = {gps->lat, gps->lon};
    z.size = 2;
    z.data = values;
    return measurement_update_internal(predicted, gps_sensor, &z, NULL, residual, kalman_gain, corrected);
}

bool camera_measurement_step(const GaussianState *predicted, const SensorModel *camera_sensor, const CameraMeasurement *camera, Residual *residual, Matrix *kalman_gain, GaussianState *corrected) {
    Vector z = {0};
    double values[2] = {
        predicted->mean.data[0] + matrix_get(&camera->pose_delta, 0, 0),
        predicted->mean.data[1] + matrix_get(&camera->pose_delta, 1, 0)
    };
    z.size = 2;
    z.data = values;
    return measurement_update_internal(predicted, camera_sensor, &z, NULL, residual, kalman_gain, corrected);
}
