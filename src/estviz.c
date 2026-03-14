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

static bool matrix_add_into(const Matrix *a, const Matrix *b, Matrix *out) {
    if (a == NULL || b == NULL || out == NULL || a->rows != b->rows || a->cols != b->cols || out->rows != a->rows || out->cols != a->cols) {
        return false;
    }
    for (int i = 0; i < a->rows * a->cols; ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

static bool matrix_sub_into(const Matrix *a, const Matrix *b, Matrix *out) {
    if (a == NULL || b == NULL || out == NULL || a->rows != b->rows || a->cols != b->cols || out->rows != a->rows || out->cols != a->cols) {
        return false;
    }
    for (int i = 0; i < a->rows * a->cols; ++i) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}

static void matrix_set_identity(Matrix *matrix) {
    memset(matrix->data, 0, sizeof(*matrix->data) * (size_t)matrix->rows * (size_t)matrix->cols);
    for (int i = 0; i < matrix->rows && i < matrix->cols; ++i) {
        matrix_set(matrix, i, i, 1.0);
    }
}

static bool vector_copy_from_matrix(Vector *vector, const Matrix *matrix) {
    if (vector == NULL || matrix == NULL || matrix->cols != 1 || vector->size != matrix->rows) {
        return false;
    }
    for (int i = 0; i < vector->size; ++i) {
        vector->data[i] = matrix_get(matrix, i, 0);
    }
    return true;
}

static bool matrix_copy_from_vector(Matrix *matrix, const Vector *vector) {
    if (matrix == NULL || vector == NULL || matrix->cols != 1 || matrix->rows != vector->size) {
        return false;
    }
    for (int i = 0; i < vector->size; ++i) {
        matrix_set(matrix, i, 0, vector->data[i]);
    }
    return true;
}

static bool gaussian_density_2d(const GaussianState *state, double x, double y, double *out_density) {
    if (state == NULL || state->mean.size != 2 || state->covariance.rows != 2 || state->covariance.cols != 2) {
        return false;
    }
    Matrix inv = {0};
    if (!matrix_init_storage(&inv, 2, 2)) {
        return false;
    }
    if (!matrix_inverse_2x2_into(&state->covariance, &inv)) {
        matrix_free_storage(&inv);
        return false;
    }
    double dx = x - state->mean.data[0];
    double dy = y - state->mean.data[1];
    double qx = matrix_get(&inv, 0, 0) * dx + matrix_get(&inv, 0, 1) * dy;
    double qy = matrix_get(&inv, 1, 0) * dx + matrix_get(&inv, 1, 1) * dy;
    double exponent = -0.5 * (dx * qx + dy * qy);
    double det = matrix_get(&state->covariance, 0, 0) * matrix_get(&state->covariance, 1, 1) -
                 matrix_get(&state->covariance, 0, 1) * matrix_get(&state->covariance, 1, 0);
    if (det <= 1e-12) det = 1e-12;
    *out_density = exp(exponent) / (2.0 * PI * sqrt(det));
    matrix_free_storage(&inv);
    return true;
}

bool gaussian_state_init(GaussianState *state, int state_size) {
    if (state == NULL) {
        return false;
    }
    memset(state, 0, sizeof(*state));
    if (!vector_init_storage(&state->mean, state_size)) {
        return false;
    }
    if (!matrix_init_storage(&state->covariance, state_size, state_size)) {
        vector_free_storage(&state->mean);
        return false;
    }
    return true;
}

void gaussian_state_free(GaussianState *state) {
    if (state == NULL) {
        return;
    }
    vector_free_storage(&state->mean);
    matrix_free_storage(&state->covariance);
}

bool measurement_init(Measurement *measurement, int z_size, int state_size) {
    if (measurement == NULL) {
        return false;
    }
    memset(measurement, 0, sizeof(*measurement));
    if (!vector_init_storage(&measurement->z, z_size)) {
        return false;
    }
    if (!matrix_init_storage(&measurement->H, z_size, state_size)) {
        vector_free_storage(&measurement->z);
        return false;
    }
    if (!matrix_init_storage(&measurement->R, z_size, z_size)) {
        vector_free_storage(&measurement->z);
        matrix_free_storage(&measurement->H);
        return false;
    }
    return true;
}

void measurement_free(Measurement *measurement) {
    if (measurement == NULL) {
        return;
    }
    vector_free_storage(&measurement->z);
    matrix_free_storage(&measurement->H);
    matrix_free_storage(&measurement->R);
}

bool residual_init(Residual *residual, int innovation_size) {
    if (residual == NULL) {
        return false;
    }
    memset(residual, 0, sizeof(*residual));
    if (!vector_init_storage(&residual->innovation, innovation_size)) {
        return false;
    }
    if (!matrix_init_storage(&residual->S, innovation_size, innovation_size)) {
        vector_free_storage(&residual->innovation);
        return false;
    }
    return true;
}

void residual_free(Residual *residual) {
    if (residual == NULL) {
        return;
    }
    vector_free_storage(&residual->innovation);
    matrix_free_storage(&residual->S);
}

bool estimation_prediction_step(const Matrix *F, const GaussianState *input, GaussianState *output) {
    Matrix in_mean = {0};
    Matrix out_mean = {0};
    bool ok = false;
    if (!matrix_init_storage(&in_mean, input->mean.size, 1) || !matrix_init_storage(&out_mean, output->mean.size, 1)) {
        goto cleanup;
    }
    matrix_copy_from_vector(&in_mean, &input->mean);
    ok = matrix_multiply_into(F, &in_mean, &out_mean) && vector_copy_from_matrix(&output->mean, &out_mean) &&
         matrix_covariance_propagate_into(F, &input->covariance, &output->covariance);
cleanup:
    matrix_free_storage(&in_mean);
    matrix_free_storage(&out_mean);
    return ok;
}

bool estimation_innovation_step(const GaussianState *predicted, const Measurement *measurement, Residual *residual) {
    Matrix x_pred = {0};
    Matrix hx = {0};
    Matrix ht = {0};
    Matrix tmp = {0};
    bool ok = false;
    if (!matrix_init_storage(&x_pred, predicted->mean.size, 1) ||
        !matrix_init_storage(&hx, measurement->z.size, 1) ||
        !matrix_init_storage(&ht, measurement->H.cols, measurement->H.rows) ||
        !matrix_init_storage(&tmp, measurement->H.rows, predicted->covariance.cols)) {
        goto cleanup;
    }
    matrix_copy_from_vector(&x_pred, &predicted->mean);
    ok = matrix_multiply_into(&measurement->H, &x_pred, &hx);
    if (!ok) goto cleanup;
    for (int i = 0; i < residual->innovation.size; ++i) {
        residual->innovation.data[i] = measurement->z.data[i] - matrix_get(&hx, i, 0);
    }
    ok = matrix_transpose_into(&measurement->H, &ht) &&
         matrix_multiply_into(&measurement->H, &predicted->covariance, &tmp) &&
         matrix_multiply_into(&tmp, &ht, &residual->S) &&
         matrix_add_into(&residual->S, &measurement->R, &residual->S);
cleanup:
    matrix_free_storage(&x_pred);
    matrix_free_storage(&hx);
    matrix_free_storage(&ht);
    matrix_free_storage(&tmp);
    return ok;
}

bool estimation_kalman_gain_step(const GaussianState *predicted, const Measurement *measurement, const Residual *residual, Matrix *kalman_gain) {
    Matrix ht = {0};
    Matrix s_inv = {0};
    Matrix tmp = {0};
    bool ok = false;
    if (!matrix_init_storage(&ht, measurement->H.cols, measurement->H.rows) ||
        !matrix_init_storage(&s_inv, residual->S.rows, residual->S.cols) ||
        !matrix_init_storage(&tmp, predicted->covariance.rows, ht.cols)) {
        goto cleanup;
    }
    ok = matrix_transpose_into(&measurement->H, &ht) &&
         matrix_inverse_2x2_into(&residual->S, &s_inv) &&
         matrix_multiply_into(&predicted->covariance, &ht, &tmp) &&
         matrix_multiply_into(&tmp, &s_inv, kalman_gain);
cleanup:
    matrix_free_storage(&ht);
    matrix_free_storage(&s_inv);
    matrix_free_storage(&tmp);
    return ok;
}

bool estimation_measurement_update_step(const GaussianState *predicted, const Residual *residual, const Measurement *measurement, const Matrix *kalman_gain, GaussianState *corrected) {
    Matrix innovation = {0};
    Matrix correction = {0};
    Matrix pred_mean = {0};
    Matrix gain_h = {0};
    Matrix identity = {0};
    Matrix tmp = {0};
    bool ok = false;
    if (!matrix_init_storage(&innovation, residual->innovation.size, 1) ||
        !matrix_init_storage(&correction, corrected->mean.size, 1) ||
        !matrix_init_storage(&pred_mean, predicted->mean.size, 1) ||
        !matrix_init_storage(&gain_h, kalman_gain->rows, measurement->H.cols) ||
        !matrix_init_storage(&identity, predicted->covariance.rows, predicted->covariance.cols) ||
        !matrix_init_storage(&tmp, predicted->covariance.rows, predicted->covariance.cols)) {
        goto cleanup;
    }
    matrix_copy_from_vector(&innovation, &residual->innovation);
    matrix_copy_from_vector(&pred_mean, &predicted->mean);
    ok = matrix_multiply_into(kalman_gain, &innovation, &correction);
    if (!ok) goto cleanup;
    for (int i = 0; i < corrected->mean.size; ++i) {
        corrected->mean.data[i] = matrix_get(&pred_mean, i, 0) + matrix_get(&correction, i, 0);
    }
    matrix_set_identity(&identity);
    ok = matrix_multiply_into(kalman_gain, &measurement->H, &gain_h) &&
         matrix_sub_into(&identity, &gain_h, &tmp) &&
         matrix_multiply_into(&tmp, &predicted->covariance, &corrected->covariance);
cleanup:
    matrix_free_storage(&innovation);
    matrix_free_storage(&correction);
    matrix_free_storage(&pred_mean);
    matrix_free_storage(&gain_h);
    matrix_free_storage(&identity);
    matrix_free_storage(&tmp);
    return ok;
}

void blueprint_draw_gaussian_state(const BlueprintEngine *engine, const GaussianState *state, Color color, const char *label) {
    if (state == NULL || state->mean.size != 2 || state->covariance.rows != 2 || state->covariance.cols != 2) {
        return;
    }
    DVec2 center = dvec2(state->mean.data[0], state->mean.data[1]);
    double lambda1 = 0.0;
    double lambda2 = 0.0;
    Vector2 e1 = {1.0f, 0.0f};
    Vector2 e2 = {0.0f, 1.0f};
    matrix_eigen_2x2(&state->covariance, &lambda1, &lambda2, &e1, &e2);
    double radius_a = sqrt(fmax(lambda1, 1e-9)) * 2.0;
    double radius_b = sqrt(fmax(lambda2, 1e-9)) * 2.0;
    double angle = atan2(e1.y, e1.x);
    const int segments = 96;
    for (int i = 0; i < segments; ++i) {
        double t0 = ((double)i / (double)segments) * 2.0 * PI;
        double t1 = ((double)(i + 1) / (double)segments) * 2.0 * PI;
        DVec2 p0 = dvec2(center.x + cos(angle) * cos(t0) * radius_a - sin(angle) * sin(t0) * radius_b,
                         center.y + sin(angle) * cos(t0) * radius_a + cos(angle) * sin(t0) * radius_b);
        DVec2 p1 = dvec2(center.x + cos(angle) * cos(t1) * radius_a - sin(angle) * sin(t1) * radius_b,
                         center.y + sin(angle) * cos(t1) * radius_a + cos(angle) * sin(t1) * radius_b);
        if (!blueprint_world_segment_visible(engine, p0, p1, 8.0)) {
            continue;
        }
        Vector2 a = blueprint_world_to_screen(engine, p0);
        Vector2 b = blueprint_world_to_screen(engine, p1);
        DrawLineEx(a, b, 2.0f, color);
    }
    blueprint_draw_arrow(engine, center, dvec2(center.x + e1.x * radius_a, center.y + e1.y * radius_a), 2.0f, color_lerp(color, WHITE, 0.2f));
    blueprint_draw_arrow(engine, center, dvec2(center.x + e2.x * radius_b, center.y + e2.y * radius_b), 2.0f, color_lerp(color, WHITE, 0.35f));
    Vector2 c = blueprint_world_to_screen(engine, center);
    DrawCircleV(c, 4.0f, color);
    if (label != NULL) {
        DrawText(label, (int)c.x + 8, (int)c.y - 10, 14, color_lerp(color, WHITE, 0.2f));
    }
}

void blueprint_draw_probability_heatmap(const BlueprintEngine *engine, const GaussianState *state, DVec2 min, DVec2 max, int steps_x, int steps_y, Color tint, const char *label) {
    double max_density = 0.0;
    for (int y = 0; y < steps_y; ++y) {
        for (int x = 0; x < steps_x; ++x) {
            double wx = min.x + ((double)x + 0.5) / (double)steps_x * (max.x - min.x);
            double wy = min.y + ((double)y + 0.5) / (double)steps_y * (max.y - min.y);
            double density = 0.0;
            if (gaussian_density_2d(state, wx, wy, &density) && density > max_density) {
                max_density = density;
            }
        }
    }
    if (max_density <= 1e-12) {
        max_density = 1.0;
    }
    double cell_w = (max.x - min.x) / (double)steps_x;
    double cell_h = (max.y - min.y) / (double)steps_y;
    for (int y = 0; y < steps_y; ++y) {
        for (int x = 0; x < steps_x; ++x) {
            double wx = min.x + ((double)x + 0.5) / (double)steps_x * (max.x - min.x);
            double wy = min.y + ((double)y + 0.5) / (double)steps_y * (max.y - min.y);
            double density = 0.0;
            if (!gaussian_density_2d(state, wx, wy, &density)) {
                continue;
            }
            float alpha = (float)(density / max_density);
            Color fill = Fade(tint, alpha * 0.65f);
            DVec2 cell_min = dvec2(min.x + x * cell_w, min.y + y * cell_h);
            DVec2 cell_max = dvec2(cell_min.x + cell_w, cell_min.y + cell_h);
            if (!blueprint_world_rect_visible(engine, cell_min, cell_max, 2.0)) {
                continue;
            }
            Vector2 a = blueprint_world_to_screen(engine, cell_min);
            Vector2 b = blueprint_world_to_screen(engine, cell_max);
            DrawRectangleV(a, (Vector2){b.x - a.x, b.y - a.y}, fill);
        }
    }
    if (label != NULL) {
        Vector2 p = blueprint_world_to_screen(engine, dvec2(min.x, min.y - 0.5));
        DrawText(label, (int)p.x, (int)p.y, 14, color_lerp(tint, WHITE, 0.2f));
    }
}

void blueprint_draw_residual_visual(const BlueprintEngine *engine, DVec2 predicted, DVec2 measured, const Residual *residual, Color color, const char *label) {
    blueprint_draw_arrow(engine, predicted, measured, 2.0f, color);
    Vector2 pred = blueprint_world_to_screen(engine, predicted);
    Vector2 meas = blueprint_world_to_screen(engine, measured);
    DrawCircleV(pred, 3.5f, color_lerp(color, WHITE, 0.15f));
    DrawCircleV(meas, 4.5f, color_lerp(color, WHITE, 0.35f));
    if (label != NULL && residual != NULL && residual->innovation.size >= 2) {
        char line[128];
        snprintf(line, sizeof(line), "%s  y=[%.2f, %.2f]", label, residual->innovation.data[0], residual->innovation.data[1]);
        DVec2 mid = dvec2((predicted.x + measured.x) * 0.5, (predicted.y + measured.y) * 0.5);
        Vector2 p = blueprint_world_to_screen(engine, mid);
        DrawText(line, (int)p.x + 8, (int)p.y - 8, 13, color);
    }
}

void blueprint_draw_uncertainty_propagation(const BlueprintEngine *engine, const GaussianState *prior, const GaussianState *propagated, Color prior_color, Color propagated_color, const char *label) {
    blueprint_draw_gaussian_state(engine, prior, prior_color, "prior");
    blueprint_draw_gaussian_state(engine, propagated, propagated_color, "propagated");
    if (prior->mean.size >= 2 && propagated->mean.size >= 2) {
        DVec2 a = dvec2(prior->mean.data[0], prior->mean.data[1]);
        DVec2 b = dvec2(propagated->mean.data[0], propagated->mean.data[1]);
        blueprint_draw_signal_arrow(engine, a, b, 1.8f, color_lerp(propagated_color, WHITE, 0.15f), 0.2);
        if (label != NULL) {
            Vector2 p = blueprint_world_to_screen(engine, dvec2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5));
            DrawText(label, (int)p.x + 8, (int)p.y + 4, 13, propagated_color);
        }
    }
}

void blueprint_draw_estimation_node_box(const BlueprintEngine *engine, const EstimationNode *node, Vector2 size, Color accent, bool active) {
    DVec2 origin = dvec2(node->world_position.x - size.x * 0.5, node->world_position.y - size.y * 0.5);
    Vector2 top_left = blueprint_world_to_screen(engine, origin);
    Rectangle rect = {top_left.x, top_left.y, blueprint_world_length_to_screen(engine, size.x), blueprint_world_length_to_screen(engine, size.y)};
    Color fill = active ? color_lerp(accent, WHITE, 0.18f) : Fade((Color){22, 28, 36, 255}, 0.95f);
    DrawRectangleRounded(rect, 0.14f, 8, fill);
    DrawRectangleRoundedLinesEx(rect, 0.14f, 8, active ? 2.4f : 1.4f, accent);
    DrawText(node->name, (int)rect.x + 10, (int)rect.y + 10, 15, (Color){236, 242, 248, 255});
}
