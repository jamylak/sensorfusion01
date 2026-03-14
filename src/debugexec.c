#include "blueprint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool simulation_frame_init(SimulationFrame *frame, int state_dim, int measurement_dim) {
    if (frame == NULL) {
        return false;
    }
    memset(frame, 0, sizeof(*frame));
    return gaussian_state_init(&frame->state, state_dim) && kalman_internals_init(&frame->internals, state_dim, measurement_dim);
}

void simulation_frame_free(SimulationFrame *frame) {
    if (frame == NULL) {
        return;
    }
    gaussian_state_free(&frame->state);
    kalman_internals_free(&frame->internals);
}

void execution_timeline_reset(ExecutionTimeline *timeline) {
    if (timeline == NULL) {
        return;
    }
    memset(timeline, 0, sizeof(*timeline));
}

bool execution_timeline_add_step(ExecutionTimeline *timeline, const char *description, void (*execute)(void), void (*draw_overlay)(void)) {
    if (timeline == NULL || timeline->step_count >= MAX_STEPS) {
        return false;
    }
    AlgorithmStep *step = &timeline->steps[timeline->step_count++];
    memset(step, 0, sizeof(*step));
    if (description != NULL) {
        strncpy(step->description, description, sizeof(step->description) - 1);
    }
    step->execute = execute;
    step->draw_overlay = draw_overlay;
    return true;
}

void execution_timeline_step_forward(ExecutionTimeline *timeline) {
    if (timeline == NULL || timeline->step_count == 0) {
        return;
    }
    if (timeline->current_step < timeline->step_count - 1) {
        timeline->current_step++;
    }
}

void execution_timeline_step_backward(ExecutionTimeline *timeline) {
    if (timeline == NULL || timeline->step_count == 0) {
        return;
    }
    if (timeline->current_step > 0) {
        timeline->current_step--;
    }
}

void blueprint_draw_execution_timeline(const BlueprintEngine *engine, const ExecutionTimeline *timeline, DVec2 origin, Vector2 size, const char *title) {
    Vector2 a = blueprint_world_to_screen(engine, origin);
    Vector2 b = blueprint_world_to_screen(engine, (DVec2){origin.x + size.x, origin.y + size.y});
    Rectangle rect = {a.x, a.y, b.x - a.x, b.y - a.y};
    DrawRectangleRounded(rect, 0.08f, 8, Fade((Color){12, 16, 22, 255}, 0.95f));
    DrawRectangleRoundedLinesEx(rect, 0.08f, 8, 1.2f, (Color){96, 112, 132, 255});
    if (title != NULL) {
        DrawText(title, (int)rect.x + 8, (int)rect.y + 8, 14, (Color){228, 236, 244, 255});
    }
    if (timeline == NULL || timeline->step_count == 0) {
        return;
    }
    float lane_x = rect.x + 20.0f;
    float lane_y = rect.y + 34.0f;
    float lane_w = rect.width - 40.0f;
    DrawLineEx((Vector2){lane_x, lane_y}, (Vector2){lane_x + lane_w, lane_y}, 2.0f, (Color){86, 102, 120, 255});
    for (int i = 0; i < timeline->step_count; ++i) {
        float t = timeline->step_count > 1 ? (float)i / (float)(timeline->step_count - 1) : 0.0f;
        float x = lane_x + t * lane_w;
        Color c = i == timeline->current_step ? (Color){244, 196, 96, 255} : (Color){118, 138, 160, 255};
        DrawCircleV((Vector2){x, lane_y}, i == timeline->current_step ? 5.5f : 4.0f, c);
        DrawText(timeline->steps[i].description, (int)rect.x + 8, (int)(rect.y + 52 + i * 18), 12, c);
    }
}

void blueprint_draw_debug_inspector(const BlueprintEngine *engine, const char *step_description, const Matrix *a, const Matrix *b, const Matrix *c, const Vector *vector, const SensorModel *sensor, DVec2 origin, float cell_size) {
    Vector2 p = blueprint_world_to_screen(engine, origin);
    DrawRectangle((int)p.x, (int)p.y, 360, 126, Fade((Color){10, 14, 18, 255}, 0.92f));
    DrawRectangleLines((int)p.x, (int)p.y, 360, 126, (Color){98, 116, 138, 255});
    DrawText("debug inspector", (int)p.x + 10, (int)p.y + 8, 15, (Color){232, 238, 246, 255});
    if (step_description != NULL) {
        DrawText(step_description, (int)p.x + 10, (int)p.y + 28, 13, (Color){244, 196, 96, 255});
    }
    char line[160];
    int y = (int)p.y + 48;
    if (a != NULL) {
        snprintf(line, sizeof(line), "A: %dx%d", a->rows, a->cols);
        DrawText(line, (int)p.x + 10, y, 12, (Color){184, 206, 232, 255});
        y += 16;
    }
    if (b != NULL) {
        snprintf(line, sizeof(line), "B: %dx%d", b->rows, b->cols);
        DrawText(line, (int)p.x + 10, y, 12, (Color){184, 206, 232, 255});
        y += 16;
    }
    if (c != NULL) {
        snprintf(line, sizeof(line), "C: %dx%d", c->rows, c->cols);
        DrawText(line, (int)p.x + 10, y, 12, (Color){184, 206, 232, 255});
        y += 16;
    }
    if (vector != NULL) {
        snprintf(line, sizeof(line), "vector size: %d", vector->size);
        DrawText(line, (int)p.x + 10, y, 12, (Color){164, 230, 194, 255});
        y += 16;
    }
    if (sensor != NULL) {
        snprintf(line, sizeof(line), "sensor: %s  zdim=%d", sensor->name, sensor->measurement_dim);
        DrawText(line, (int)p.x + 10, y, 12, (Color){246, 178, 102, 255});
    }

    if (a != NULL) blueprint_draw_matrix_heatmap(engine, a, (DVec2){origin.x + 4.0, origin.y + 8.6}, cell_size, true, -1, -1, -1, -1, NULL);
    if (b != NULL) blueprint_draw_matrix_heatmap(engine, b, (DVec2){origin.x + 8.2, origin.y + 8.6}, cell_size, true, -1, -1, -1, -1, NULL);
    if (c != NULL) blueprint_draw_matrix_heatmap(engine, c, (DVec2){origin.x + 12.4, origin.y + 8.6}, cell_size, true, -1, -1, -1, -1, NULL);
}
