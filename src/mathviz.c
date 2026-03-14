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

static Vector2 normalize_vector2(Vector2 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    if (len <= 1e-6f) {
        return (Vector2){1.0f, 0.0f};
    }
    return (Vector2){v.x / len, v.y / len};
}

static double matrix_abs_max(const Matrix *matrix) {
    double max_value = 1.0;
    for (int i = 0; i < matrix->rows * matrix->cols; ++i) {
        double value = fabs(matrix->data[i]);
        if (value > max_value) {
            max_value = value;
        }
    }
    return max_value;
}

static Color matrix_value_color(double value, double scale) {
    const Color neg = (Color){48, 116, 214, 255};
    const Color pos = (Color){234, 122, 52, 255};
    const Color mid = (Color){30, 36, 48, 255};
    float t = (float)((value / scale + 1.0) * 0.5);
    Color blend = t < 0.5f ? color_lerp(neg, mid, t * 2.0f) : color_lerp(mid, pos, (t - 0.5f) * 2.0f);
    return blend;
}

Matrix *matrix_create(int rows, int cols) {
    Matrix *matrix = calloc(1, sizeof(*matrix));
    if (matrix == NULL) {
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = calloc((size_t)rows * (size_t)cols, sizeof(*matrix->data));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
    return matrix;
}

bool matrix_init_storage(Matrix *matrix, int rows, int cols) {
    if (matrix == NULL) {
        return false;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = calloc((size_t)rows * (size_t)cols, sizeof(*matrix->data));
    return matrix->data != NULL;
}

Matrix *matrix_from_array(int rows, int cols, const double *values) {
    Matrix *matrix = matrix_create(rows, cols);
    if (matrix == NULL) {
        return NULL;
    }
    memcpy(matrix->data, values, sizeof(*matrix->data) * (size_t)rows * (size_t)cols);
    return matrix;
}

void matrix_destroy(Matrix *matrix) {
    if (matrix == NULL) {
        return;
    }
    free(matrix->data);
    free(matrix);
}

void matrix_free_storage(Matrix *matrix) {
    if (matrix == NULL) {
        return;
    }
    free(matrix->data);
    matrix->data = NULL;
    matrix->rows = 0;
    matrix->cols = 0;
}

double matrix_get(const Matrix *matrix, int row, int col) {
    return matrix->data[row * matrix->cols + col];
}

void matrix_set(Matrix *matrix, int row, int col, double value) {
    matrix->data[row * matrix->cols + col] = value;
}

bool matrix_multiply_into(const Matrix *a, const Matrix *b, Matrix *out) {
    if (a == NULL || b == NULL || out == NULL || a->cols != b->rows || out->rows != a->rows || out->cols != b->cols) {
        return false;
    }
    for (int r = 0; r < out->rows; ++r) {
        for (int c = 0; c < out->cols; ++c) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; ++k) {
                sum += matrix_get(a, r, k) * matrix_get(b, k, c);
            }
            matrix_set(out, r, c, sum);
        }
    }
    return true;
}

bool matrix_transpose_into(const Matrix *in, Matrix *out) {
    if (in == NULL || out == NULL || out->rows != in->cols || out->cols != in->rows) {
        return false;
    }
    for (int r = 0; r < in->rows; ++r) {
        for (int c = 0; c < in->cols; ++c) {
            matrix_set(out, c, r, matrix_get(in, r, c));
        }
    }
    return true;
}

bool matrix_inverse_2x2_into(const Matrix *in, Matrix *out) {
    if (in == NULL || out == NULL || in->rows != 2 || in->cols != 2 || out->rows != 2 || out->cols != 2) {
        return false;
    }
    double a = matrix_get(in, 0, 0);
    double b = matrix_get(in, 0, 1);
    double c = matrix_get(in, 1, 0);
    double d = matrix_get(in, 1, 1);
    double det = a * d - b * c;
    if (fabs(det) <= 1e-12) {
        return false;
    }
    matrix_set(out, 0, 0, d / det);
    matrix_set(out, 0, 1, -b / det);
    matrix_set(out, 1, 0, -c / det);
    matrix_set(out, 1, 1, a / det);
    return true;
}

bool matrix_covariance_propagate_into(const Matrix *f, const Matrix *cov, Matrix *out) {
    if (f == NULL || cov == NULL || out == NULL || f->rows != f->cols || cov->rows != cov->cols || f->cols != cov->rows ||
        out->rows != cov->rows || out->cols != cov->cols) {
        return false;
    }
    Matrix *tmp = matrix_create(f->rows, cov->cols);
    Matrix *ft = matrix_create(f->cols, f->rows);
    if (tmp == NULL || ft == NULL) {
        matrix_destroy(tmp);
        matrix_destroy(ft);
        return false;
    }
    bool ok = matrix_multiply_into(f, cov, tmp) && matrix_transpose_into(f, ft) && matrix_multiply_into(tmp, ft, out);
    matrix_destroy(tmp);
    matrix_destroy(ft);
    return ok;
}

bool matrix_eigen_2x2(const Matrix *in, double *lambda1, double *lambda2, Vector2 *eigenvector1, Vector2 *eigenvector2) {
    if (in == NULL || in->rows != 2 || in->cols != 2) {
        return false;
    }
    double a = matrix_get(in, 0, 0);
    double b = matrix_get(in, 0, 1);
    double c = matrix_get(in, 1, 0);
    double d = matrix_get(in, 1, 1);
    double trace = a + d;
    double discriminant = trace * trace - 4.0 * (a * d - b * c);
    if (discriminant < 0.0) {
        discriminant = 0.0;
    }
    double root = sqrt(discriminant);
    *lambda1 = 0.5 * (trace + root);
    *lambda2 = 0.5 * (trace - root);

    Vector2 v1 = fabs(b) > 1e-9 ? (Vector2){(float)(*lambda1 - d), (float)b} : (Vector2){(float)c, (float)(*lambda1 - a)};
    Vector2 v2 = fabs(b) > 1e-9 ? (Vector2){(float)(*lambda2 - d), (float)b} : (Vector2){(float)c, (float)(*lambda2 - a)};
    if (fabsf(v1.x) + fabsf(v1.y) < 1e-6f) v1 = (Vector2){1.0f, 0.0f};
    if (fabsf(v2.x) + fabsf(v2.y) < 1e-6f) v2 = (Vector2){0.0f, 1.0f};
    *eigenvector1 = normalize_vector2(v1);
    *eigenvector2 = normalize_vector2(v2);
    return true;
}

Vector *vector_create(int size) {
    Vector *vector = calloc(1, sizeof(*vector));
    if (vector == NULL) {
        return NULL;
    }
    vector->size = size;
    vector->data = calloc((size_t)size, sizeof(*vector->data));
    if (vector->data == NULL) {
        free(vector);
        return NULL;
    }
    return vector;
}

bool vector_init_storage(Vector *vector, int size) {
    if (vector == NULL) {
        return false;
    }
    vector->size = size;
    vector->data = calloc((size_t)size, sizeof(*vector->data));
    return vector->data != NULL;
}

Vector *vector_from_array(int size, const double *values) {
    Vector *vector = vector_create(size);
    if (vector == NULL) {
        return NULL;
    }
    memcpy(vector->data, values, sizeof(*vector->data) * (size_t)size);
    return vector;
}

void vector_destroy(Vector *vector) {
    if (vector == NULL) {
        return;
    }
    free(vector->data);
    free(vector);
}

void vector_free_storage(Vector *vector) {
    if (vector == NULL) {
        return;
    }
    free(vector->data);
    vector->data = NULL;
    vector->size = 0;
}

bool matrix_transform_vector_into(const Matrix *matrix, const Vector *vector, Vector *out) {
    if (matrix == NULL || vector == NULL || out == NULL || matrix->cols != vector->size || matrix->rows != out->size) {
        return false;
    }
    for (int r = 0; r < matrix->rows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < matrix->cols; ++c) {
            sum += matrix_get(matrix, r, c) * vector->data[c];
        }
        out->data[r] = sum;
    }
    return true;
}

void blueprint_draw_matrix_heatmap(const BlueprintEngine *engine, const Matrix *matrix, DVec2 origin, float cell_size, bool show_values, int highlight_row, int highlight_col, int focus_row, int focus_col, const char *title) {
    double scale = matrix_abs_max(matrix);
    float screen_cell = blueprint_world_length_to_screen(engine, cell_size);
    for (int r = 0; r < matrix->rows; ++r) {
        for (int c = 0; c < matrix->cols; ++c) {
            DVec2 cell_origin = dvec2(origin.x + c * cell_size, origin.y + r * cell_size);
            if (!blueprint_world_rect_visible(engine, cell_origin, dvec2(cell_origin.x + cell_size, cell_origin.y + cell_size), 4.0)) {
                continue;
            }
            double value = matrix_get(matrix, r, c);
            Color fill = matrix_value_color(value, scale);
            if (r == highlight_row || c == highlight_col) {
                fill = color_lerp(fill, (Color){236, 234, 181, 255}, 0.30f);
            }
            if (r == focus_row && c == focus_col) {
                fill = color_lerp(fill, (Color){255, 245, 160, 255}, 0.55f);
            }
            Vector2 min = blueprint_world_to_screen(engine, cell_origin);
            Vector2 max = blueprint_world_to_screen(engine, dvec2(cell_origin.x + cell_size, cell_origin.y + cell_size));
            DrawRectangleV(min, (Vector2){max.x - min.x, max.y - min.y}, fill);
            if (show_values && screen_cell >= 24.0f) {
                char text[32];
                snprintf(text, sizeof(text), "%.2f", value);
                int font_size = (int)(screen_cell * 0.28f);
                if (font_size < 10) font_size = 10;
                DrawText(text, (int)min.x + 6, (int)min.y + (int)(screen_cell * 0.35f), font_size, (Color){235, 242, 250, 255});
            }
        }
    }
    blueprint_draw_matrix_grid(engine, origin, matrix->rows, matrix->cols, cell_size, (Color){190, 212, 236, 255}, 1.1f);
    if (title != NULL) {
        Vector2 anchor = blueprint_world_to_screen(engine, dvec2(origin.x, origin.y - cell_size * 0.8));
        DrawText(title, (int)anchor.x, (int)anchor.y, 16, (Color){228, 236, 244, 255});
    }
}

void blueprint_draw_vector_visual(const BlueprintEngine *engine, const Vector *vector, DVec2 origin, float cell_size, bool show_values, const char *title, Color accent) {
    Matrix wrapper = {vector->size, 1, vector->data};
    blueprint_draw_matrix_heatmap(engine, &wrapper, origin, cell_size, show_values, -1, -1, -1, -1, title);

    double abs_max = 1.0;
    for (int i = 0; i < vector->size; ++i) {
        if (fabs(vector->data[i]) > abs_max) {
            abs_max = fabs(vector->data[i]);
        }
    }

    DVec2 zero_axis_start = dvec2(origin.x + cell_size * 2.0, origin.y);
    DVec2 zero_axis_end = dvec2(origin.x + cell_size * 2.0, origin.y + vector->size * cell_size);
    Vector2 za = blueprint_world_to_screen(engine, zero_axis_start);
    Vector2 zb = blueprint_world_to_screen(engine, zero_axis_end);
    DrawLineEx(za, zb, 1.0f, Fade(accent, 0.55f));

    for (int i = 0; i < vector->size; ++i) {
        double magnitude = vector->data[i] / abs_max;
        DVec2 from = dvec2(origin.x + cell_size * 2.0, origin.y + (i + 0.5) * cell_size);
        DVec2 to = dvec2(from.x + magnitude * cell_size * 1.8, from.y);
        blueprint_draw_arrow(engine, from, to, 1.6f, accent);
    }
}

void blueprint_draw_tensor_heatmap(const BlueprintEngine *engine, const TensorHeatmap *heatmap, bool show_values, const char *title) {
    blueprint_draw_matrix_heatmap(engine, heatmap->matrix, dvec2(heatmap->world_position.x, heatmap->world_position.y), heatmap->cell_size, show_values, -1, -1, -1, -1, title);
}

void blueprint_draw_math_node_box(const BlueprintEngine *engine, const MathNode *node, Vector2 size, Color accent, bool active) {
    DVec2 origin = dvec2(node->position.x - size.x * 0.5, node->position.y - size.y * 0.5);
    Vector2 top_left = blueprint_world_to_screen(engine, origin);
    float width = blueprint_world_length_to_screen(engine, size.x);
    float height = blueprint_world_length_to_screen(engine, size.y);
    Rectangle rect = {top_left.x, top_left.y, width, height};
    Color fill = active ? color_lerp(accent, WHITE, 0.18f) : Fade((Color){22, 28, 36, 255}, 0.96f);
    DrawRectangleRounded(rect, 0.12f, 8, fill);
    DrawRectangleRoundedLinesEx(rect, 0.12f, 8, active ? 2.4f : 1.4f, accent);
    DrawText(node->name, (int)rect.x + 10, (int)rect.y + 10, 16, (Color){232, 238, 246, 255});
    if (node->output != NULL) {
        char dims[32];
        snprintf(dims, sizeof(dims), "%dx%d", node->output->rows, node->output->cols);
        DrawText(dims, (int)rect.x + 10, (int)rect.y + 32, 13, Fade((Color){224, 232, 244, 255}, 0.8f));
    }
}

void blueprint_draw_tensor_flow_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, Color color, const char *label, bool active) {
    if (active) {
        blueprint_draw_signal_arrow(engine, from, to, 2.3f, color, 0.0);
    } else {
        blueprint_draw_arrow(engine, from, to, 1.6f, Fade(color, 0.85f));
    }
    if (label != NULL) {
        DVec2 mid = dvec2((from.x + to.x) * 0.5, (from.y + to.y) * 0.5);
        Vector2 p = blueprint_world_to_screen(engine, mid);
        DrawText(label, (int)p.x + 6, (int)p.y - 6, 13, color_lerp(color, WHITE, 0.3f));
    }
}

void blueprint_draw_matrix_multiply_visualizer(const BlueprintEngine *engine, const Matrix *a, const Matrix *b, const Matrix *c, DVec2 origin_a, DVec2 origin_b, DVec2 origin_c, float cell_size, double time_seconds) {
    int total_outputs = c->rows * c->cols;
    int total_steps = total_outputs * a->cols;
    int step = (int)floor(time_seconds * 1.8) % (total_steps > 0 ? total_steps : 1);
    int output_index = step / a->cols;
    int k = step % a->cols;
    int i = output_index / c->cols;
    int j = output_index % c->cols;

    blueprint_draw_matrix_heatmap(engine, a, origin_a, cell_size, true, i, -1, i, k, "A");
    blueprint_draw_matrix_heatmap(engine, b, origin_b, cell_size, true, -1, j, k, j, "B");
    blueprint_draw_matrix_heatmap(engine, c, origin_c, cell_size, true, i, j, i, j, "C = A x B");

    DVec2 row_anchor = dvec2(origin_a.x + a->cols * cell_size + cell_size * 0.2, origin_a.y + (i + 0.5) * cell_size);
    DVec2 col_anchor = dvec2(origin_b.x - cell_size * 0.2, origin_b.y + (k + 0.5) * cell_size);
    DVec2 out_anchor = dvec2(origin_c.x - cell_size * 0.25, origin_c.y + (i + 0.5) * cell_size);
    blueprint_draw_signal_arrow(engine, row_anchor, col_anchor, 2.0f, (Color){244, 164, 84, 240}, 0.1);
    blueprint_draw_signal_arrow(engine, col_anchor, out_anchor, 2.0f, (Color){116, 228, 186, 230}, 0.32);

    double partial = 0.0;
    for (int idx = 0; idx <= k; ++idx) {
        partial += matrix_get(a, i, idx) * matrix_get(b, idx, j);
    }
    char line0[128];
    char line1[128];
    char line2[128];
    snprintf(line0, sizeof(line0), "active output: C[%d,%d]", i, j);
    snprintf(line1, sizeof(line1), "term k=%d: %.2f x %.2f", k, matrix_get(a, i, k), matrix_get(b, k, j));
    snprintf(line2, sizeof(line2), "partial sum: %.3f / final %.3f", partial, matrix_get(c, i, j));
    const char *lines[] = {line0, line1, line2};
    blueprint_draw_equation_block(engine, dvec2(origin_b.x + b->cols * cell_size + 60.0, origin_b.y - cell_size * 0.5), "row(A) . column(B)", lines, 3, (Color){236, 243, 248, 255});
}

void blueprint_draw_covariance_matrix_visual(const BlueprintEngine *engine, const Matrix *covariance, DVec2 matrix_origin, DVec2 ellipse_origin, float cell_size, const char *title) {
    double lambda1 = 0.0;
    double lambda2 = 0.0;
    Vector2 eig1 = {1.0f, 0.0f};
    Vector2 eig2 = {0.0f, 1.0f};
    matrix_eigen_2x2(covariance, &lambda1, &lambda2, &eig1, &eig2);

    blueprint_draw_matrix_heatmap(engine, covariance, matrix_origin, cell_size, true, -1, -1, -1, -1, title);

    double radius_a = sqrt(fmax(lambda1, 1e-9)) * 54.0;
    double radius_b = sqrt(fmax(lambda2, 1e-9)) * 54.0;
    double angle = atan2(eig1.y, eig1.x);
    const int segments = 96;
    DVec2 points[96];
    for (int i = 0; i < segments; ++i) {
        double t = ((double)i / (double)segments) * 2.0 * PI;
        double ex = cos(t) * radius_a;
        double ey = sin(t) * radius_b;
        double rx = cos(angle) * ex - sin(angle) * ey;
        double ry = sin(angle) * ex + cos(angle) * ey;
        points[i] = dvec2(ellipse_origin.x + rx, ellipse_origin.y + ry);
    }
    for (int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;
        if (!blueprint_world_segment_visible(engine, points[i], points[next], 24.0)) {
            continue;
        }
        Vector2 a = blueprint_world_to_screen(engine, points[i]);
        Vector2 b = blueprint_world_to_screen(engine, points[next]);
        DrawLineEx(a, b, 2.0f, (Color){188, 126, 255, 255});
    }
    blueprint_draw_arrow(engine, ellipse_origin, dvec2(ellipse_origin.x + eig1.x * radius_a, ellipse_origin.y + eig1.y * radius_a), 2.0f, (Color){255, 214, 146, 255});
    blueprint_draw_arrow(engine, ellipse_origin, dvec2(ellipse_origin.x + eig2.x * radius_b, ellipse_origin.y + eig2.y * radius_b), 2.0f, (Color){132, 234, 224, 255});

    char line0[128];
    char line1[128];
    snprintf(line0, sizeof(line0), "lambda1=%.3f  v1=[%.2f, %.2f]", lambda1, eig1.x, eig1.y);
    snprintf(line1, sizeof(line1), "lambda2=%.3f  v2=[%.2f, %.2f]", lambda2, eig2.x, eig2.y);
    const char *lines[] = {line0, line1};
    blueprint_draw_equation_block(engine, dvec2(ellipse_origin.x - 140.0, ellipse_origin.y - 150.0), "eig(cov)", lines, 2, (Color){238, 226, 255, 255});
}
