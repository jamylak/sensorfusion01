#include "blueprint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static BlueprintEngine *g_engine = NULL;
static BlueprintNode *g_active_node = NULL;

static DVec2 dvec2(double x, double y) {
    DVec2 v = {x, y};
    return v;
}

static DVec2 dvec2_add(DVec2 a, DVec2 b) {
    return dvec2(a.x + b.x, a.y + b.y);
}

static DVec2 dvec2_sub(DVec2 a, DVec2 b) {
    return dvec2(a.x - b.x, a.y - b.y);
}

static DVec2 dvec2_scale(DVec2 v, double s) {
    return dvec2(v.x * s, v.y * s);
}

static double dvec2_length(DVec2 v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

static DVec2 dvec2_normalize(DVec2 v) {
    double len = dvec2_length(v);
    if (len <= 1e-12) {
        return dvec2(0.0, 0.0);
    }
    return dvec2(v.x / len, v.y / len);
}

static DVec2 dvec2_perp(DVec2 v) {
    return dvec2(-v.y, v.x);
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

static float clampf(float value, float min_value, float max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static double clampd(double value, double min_value, double max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static DVec2 viewport_center(const BlueprintEngine *engine) {
    return dvec2(engine->camera.viewport_width * 0.5, engine->camera.top_bar_height + engine->camera.viewport_height * 0.5);
}

static Rectangle minimap_rect(const BlueprintEngine *engine) {
    float width = 220.0f;
    float height = 160.0f;
    float margin = 16.0f;
    return (Rectangle){
        engine->camera.viewport_width - width - margin,
        engine->camera.top_bar_height + engine->camera.viewport_height - height - margin,
        width,
        height
    };
}

static void get_world_view_bounds(const BlueprintEngine *engine, DVec2 *out_min, DVec2 *out_max) {
    DVec2 top_left = blueprint_screen_to_world(engine, (Vector2){0.0f, engine->camera.top_bar_height});
    DVec2 bottom_right = blueprint_screen_to_world(engine, (Vector2){engine->camera.viewport_width, engine->camera.top_bar_height + engine->camera.viewport_height});
    out_min->x = fmin(top_left.x, bottom_right.x);
    out_min->y = fmin(top_left.y, bottom_right.y);
    out_max->x = fmax(top_left.x, bottom_right.x);
    out_max->y = fmax(top_left.y, bottom_right.y);
}

static void get_world_content_bounds(const BlueprintEngine *engine, DVec2 *out_min, DVec2 *out_max) {
    DVec2 min = dvec2(-1000.0, -1000.0);
    DVec2 max = dvec2(1000.0, 1000.0);
    bool found = false;
    for (size_t i = 0; i < engine->count; ++i) {
        const BlueprintNode *node = &engine->nodes[i];
        if (!node->visible || node->page != engine->active_page) {
            continue;
        }
        if (!found) {
            min = node->bounds_min;
            max = node->bounds_max;
            found = true;
            continue;
        }
        min.x = fmin(min.x, node->bounds_min.x);
        min.y = fmin(min.y, node->bounds_min.y);
        max.x = fmax(max.x, node->bounds_max.x);
        max.y = fmax(max.y, node->bounds_max.y);
    }
    double pad_x = fmax((max.x - min.x) * 0.08, 120.0);
    double pad_y = fmax((max.y - min.y) * 0.08, 120.0);
    out_min->x = min.x - pad_x;
    out_min->y = min.y - pad_y;
    out_max->x = max.x + pad_x;
    out_max->y = max.y + pad_y;
}

static Vector2 minimap_world_to_screen(Rectangle map_rect, DVec2 world_min, DVec2 world_max, DVec2 point) {
    double width = fmax(world_max.x - world_min.x, 1.0);
    double height = fmax(world_max.y - world_min.y, 1.0);
    float x = map_rect.x + (float)(((point.x - world_min.x) / width) * map_rect.width);
    float y = map_rect.y + (float)(((point.y - world_min.y) / height) * map_rect.height);
    return (Vector2){x, y};
}

static DVec2 minimap_screen_to_world(Rectangle map_rect, DVec2 world_min, DVec2 world_max, Vector2 point) {
    double tx = clampd((point.x - map_rect.x) / map_rect.width, 0.0, 1.0);
    double ty = clampd((point.y - map_rect.y) / map_rect.height, 0.0, 1.0);
    return dvec2(
        world_min.x + (world_max.x - world_min.x) * tx,
        world_min.y + (world_max.y - world_min.y) * ty
    );
}

static void handle_minimap_input(BlueprintEngine *engine, Vector2 mouse) {
    Rectangle map_rect = minimap_rect(engine);
    if (!CheckCollisionPointRec(mouse, map_rect)) {
        return;
    }
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        return;
    }

    DVec2 world_min;
    DVec2 world_max;
    get_world_content_bounds(engine, &world_min, &world_max);
    DVec2 target = minimap_screen_to_world(map_rect, world_min, world_max, mouse);
    engine->camera.target_goal_x = target.x;
    engine->camera.target_goal_y = target.y;
}

static void draw_minimap(const BlueprintEngine *engine) {
    Rectangle map_rect = minimap_rect(engine);
    DVec2 world_min;
    DVec2 world_max;
    DVec2 view_min;
    DVec2 view_max;
    get_world_content_bounds(engine, &world_min, &world_max);
    get_world_view_bounds(engine, &view_min, &view_max);

    DrawRectangleRounded(map_rect, 0.06f, 8, Fade((Color){10, 14, 20, 255}, 0.94f));
    DrawRectangleRoundedLinesEx(map_rect, 0.06f, 8, 1.5f, (Color){84, 102, 128, 255});

    for (size_t i = 0; i < engine->count; ++i) {
        const BlueprintNode *node = &engine->nodes[i];
        Vector2 a = minimap_world_to_screen(map_rect, world_min, world_max, node->bounds_min);
        Vector2 b = minimap_world_to_screen(map_rect, world_min, world_max, node->bounds_max);
        float x = fminf(a.x, b.x);
        float y = fminf(a.y, b.y);
        float w = fabsf(b.x - a.x);
        float h = fabsf(b.y - a.y);
        if (w < 2.0f) w = 2.0f;
        if (h < 2.0f) h = 2.0f;
        if (x + w > map_rect.x + map_rect.width) w = map_rect.x + map_rect.width - x;
        if (y + h > map_rect.y + map_rect.height) h = map_rect.y + map_rect.height - y;
        DrawRectangleLinesEx((Rectangle){x, y, w, h}, 1.0f, (Color){102, 138, 178, 180});
    }

    Vector2 va = minimap_world_to_screen(map_rect, world_min, world_max, view_min);
    Vector2 vb = minimap_world_to_screen(map_rect, world_min, world_max, view_max);
    float vx = clampf(fminf(va.x, vb.x), map_rect.x, map_rect.x + map_rect.width);
    float vy = clampf(fminf(va.y, vb.y), map_rect.y, map_rect.y + map_rect.height);
    float vw = fabsf(vb.x - va.x);
    float vh = fabsf(vb.y - va.y);
    if (vw > map_rect.width) vw = map_rect.width;
    if (vh > map_rect.height) vh = map_rect.height;
    if (vx + vw > map_rect.x + map_rect.width) vx = map_rect.x + map_rect.width - vw;
    if (vy + vh > map_rect.y + map_rect.height) vy = map_rect.y + map_rect.height - vh;
    DrawRectangleLinesEx((Rectangle){vx, vy, vw, vh}, 2.0f, (Color){244, 196, 96, 255});

    Vector2 center = minimap_world_to_screen(map_rect, world_min, world_max, dvec2(engine->camera.target_x, engine->camera.target_y));
    center.x = clampf(center.x, map_rect.x, map_rect.x + map_rect.width);
    center.y = clampf(center.y, map_rect.y, map_rect.y + map_rect.height);
    DrawCircleV(center, 3.0f, (Color){244, 196, 96, 255});
    DrawText("map", (int)map_rect.x + 8, (int)map_rect.y + 6, 12, (Color){206, 218, 235, 255});
}

Camera2D blueprint_camera_snapshot(const BlueprintEngine *engine) {
    DVec2 center = viewport_center(engine);
    Camera2D cam = {0};
    cam.offset = (Vector2){(float)center.x, (float)center.y};
    cam.target = (Vector2){(float)engine->camera.target_x, (float)engine->camera.target_y};
    cam.rotation = 0.0f;
    cam.zoom = (float)engine->camera.zoom;
    return cam;
}

const BlueprintEngine *blueprint_active_engine(void) {
    return g_engine;
}

const BlueprintNode *blueprint_active_node(void) {
    return g_active_node;
}

bool blueprint_world_rect_visible(const BlueprintEngine *engine, DVec2 min, DVec2 max, double padding) {
    DVec2 view_min;
    DVec2 view_max;
    get_world_view_bounds(engine, &view_min, &view_max);
    return !(max.x < view_min.x - padding ||
             min.x > view_max.x + padding ||
             max.y < view_min.y - padding ||
             min.y > view_max.y + padding);
}

bool blueprint_world_segment_visible(const BlueprintEngine *engine, DVec2 a, DVec2 b, double padding) {
    DVec2 min = dvec2(fmin(a.x, b.x), fmin(a.y, b.y));
    DVec2 max = dvec2(fmax(a.x, b.x), fmax(a.y, b.y));
    return blueprint_world_rect_visible(engine, min, max, padding);
}

bool blueprint_world_point_visible(const BlueprintEngine *engine, DVec2 point, double padding) {
    return blueprint_world_rect_visible(engine, point, point, padding);
}

DVec2 blueprint_node_origin(void) {
    if (g_active_node == NULL) {
        return dvec2(0.0, 0.0);
    }
    return g_active_node->precise_world_position;
}

DVec2 blueprint_screen_to_world(const BlueprintEngine *engine, Vector2 screen) {
    DVec2 center = viewport_center(engine);
    double dx = ((double)screen.x - center.x) / engine->camera.zoom;
    double dy = ((double)screen.y - center.y) / engine->camera.zoom;
    return dvec2(engine->camera.target_x + dx, engine->camera.target_y + dy);
}

Vector2 blueprint_world_to_screen(const BlueprintEngine *engine, DVec2 world) {
    DVec2 center = viewport_center(engine);
    float sx = (float)((world.x - engine->camera.target_x) * engine->camera.zoom + center.x);
    float sy = (float)((world.y - engine->camera.target_y) * engine->camera.zoom + center.y);
    return (Vector2){sx, sy};
}

float blueprint_world_length_to_screen(const BlueprintEngine *engine, double length) {
    return (float)(length * engine->camera.zoom);
}

static void draw_world_polyline(const BlueprintEngine *engine, const DVec2 *points, int count, float thickness, Color color, bool closed) {
    for (int i = 0; i < count - 1; ++i) {
        Vector2 a = blueprint_world_to_screen(engine, points[i]);
        Vector2 b = blueprint_world_to_screen(engine, points[i + 1]);
        DrawLineEx(a, b, thickness, color);
    }
    if (closed && count > 2) {
        Vector2 a = blueprint_world_to_screen(engine, points[count - 1]);
        Vector2 b = blueprint_world_to_screen(engine, points[0]);
        DrawLineEx(a, b, thickness, color);
    }
}

void blueprint_engine_init(BlueprintEngine *engine, int width, int height) {
    memset(engine, 0, sizeof(*engine));
    engine->camera.top_bar_height = 44.0f;
    blueprint_engine_reset(engine, width, height);
}

void blueprint_engine_shutdown(BlueprintEngine *engine) {
    free(engine->nodes);
    engine->nodes = NULL;
    engine->count = 0;
    engine->capacity = 0;
}

void blueprint_engine_reset(BlueprintEngine *engine, int width, int height) {
    engine->active_page = 0;
    engine->paused = false;
    engine->quit_requested = false;
    engine->time_seconds = 0.0;
    engine->signal_phase = 0.0;
    engine->camera.target_x = 0.0;
    engine->camera.target_y = 0.0;
    engine->camera.target_goal_x = 0.0;
    engine->camera.target_goal_y = 0.0;
    engine->camera.zoom = 0.20;
    engine->camera.zoom_goal = 0.20;
    blueprint_engine_set_viewport(engine, width, height);
}

void blueprint_engine_set_viewport(BlueprintEngine *engine, int width, int height) {
    engine->camera.viewport_width = (float)width;
    engine->camera.viewport_height = (float)height - engine->camera.top_bar_height;
    if (engine->camera.viewport_height < 1.0f) {
        engine->camera.viewport_height = 1.0f;
    }
}

void blueprint_engine_add_node(BlueprintEngine *engine, const BlueprintNode *node) {
    if (engine->count == engine->capacity) {
        size_t new_capacity = engine->capacity == 0 ? 8 : engine->capacity * 2;
        BlueprintNode *new_nodes = realloc(engine->nodes, new_capacity * sizeof(*new_nodes));
        if (new_nodes == NULL) {
            fprintf(stderr, "failed to grow node registry\n");
            exit(1);
        }
        engine->nodes = new_nodes;
        engine->capacity = new_capacity;
    }
    engine->nodes[engine->count++] = *node;
}

static void draw_top_bar(BlueprintEngine *engine) {
    const Color bar_color = (Color){12, 18, 28, 255};
    const Color border_color = (Color){58, 78, 110, 255};
    const Color active = (Color){170, 205, 255, 255};
    const Color inactive = (Color){120, 138, 164, 255};
    DrawRectangle(0, 0, GetScreenWidth(), (int)engine->camera.top_bar_height, bar_color);
    DrawLine(0, (int)engine->camera.top_bar_height, GetScreenWidth(), (int)engine->camera.top_bar_height, border_color);

    Rectangle tab1 = {12, 8, 54, 28};
    Rectangle tab2 = {72, 8, 54, 28};
    Vector2 mouse = GetMousePosition();

    if (CheckCollisionPointRec(mouse, tab1) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        engine->active_page = 0;
    }
    if (CheckCollisionPointRec(mouse, tab2) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        engine->active_page = 1;
    }

    DrawRectangleRounded(tab1, 0.2f, 4, engine->active_page == 0 ? active : inactive);
    DrawRectangleRounded(tab2, 0.2f, 4, engine->active_page == 1 ? active : inactive);
    DrawText("[1]", 24, 15, 14, engine->active_page == 0 ? bar_color : BLACK);
    DrawText("[2]", 84, 15, 14, engine->active_page == 1 ? bar_color : BLACK);

    DrawText("Space pause/resume  R reset  Q quit  Drag pan  Wheel zoom  HJKL nudge", 148, 15, 14, (Color){205, 215, 230, 255});
}

static void draw_debug_overlay(const BlueprintEngine *engine) {
    char info[256];
    snprintf(info, sizeof(info), "page=%d  nodes=%zu  zoom=%.5f  target=(%.3f, %.3f)  paused=%s",
             engine->active_page + 1, engine->count, engine->camera.zoom,
             engine->camera.target_x, engine->camera.target_y,
             engine->paused ? "yes" : "no");
    DrawRectangle(12, GetScreenHeight() - 30, 520, 18, Fade(BLACK, 0.5f));
    DrawText(info, 18, GetScreenHeight() - 28, 14, (Color){220, 230, 240, 255});
}

static void handle_camera_input(BlueprintEngine *engine) {
    Vector2 mouse = GetMousePosition();
    bool mouse_in_view = mouse.y >= engine->camera.top_bar_height;
    bool mouse_in_minimap = CheckCollisionPointRec(mouse, minimap_rect(engine));

    if (IsKeyPressed(KEY_ONE)) engine->active_page = 0;
    if (IsKeyPressed(KEY_TWO)) engine->active_page = 1;
    if (IsKeyPressed(KEY_SPACE)) engine->paused = !engine->paused;
    if (IsKeyPressed(KEY_R)) blueprint_engine_reset(engine, GetScreenWidth(), GetScreenHeight());
    if (IsKeyPressed(KEY_Q)) engine->quit_requested = true;

    if (engine->active_page == 0) {
        handle_minimap_input(engine, mouse);
    }

    double key_pan = 44.0 / engine->camera.zoom_goal;
    if (IsKeyDown(KEY_H)) engine->camera.target_goal_x -= key_pan;
    if (IsKeyDown(KEY_L)) engine->camera.target_goal_x += key_pan;
    if (IsKeyDown(KEY_K)) engine->camera.target_goal_y -= key_pan;
    if (IsKeyDown(KEY_J)) engine->camera.target_goal_y += key_pan;

    float wheel = mouse_in_view ? GetMouseWheelMove() : 0.0f;
    if (wheel != 0.0f) {
        double old_zoom = engine->camera.zoom_goal;
        DVec2 before = blueprint_screen_to_world(engine, mouse);
        double zoom_factor = pow(1.18, wheel);
        engine->camera.zoom_goal = clampd(engine->camera.zoom_goal * zoom_factor, 0.00002, 250.0);
        DVec2 center = viewport_center(engine);
        engine->camera.target_goal_x = before.x - (((double)mouse.x - center.x) / engine->camera.zoom_goal);
        engine->camera.target_goal_y = before.y - (((double)mouse.y - center.y) / engine->camera.zoom_goal);
        if (old_zoom == engine->camera.zoom_goal) {
            engine->camera.zoom_goal = old_zoom;
        }
    }

    if (mouse_in_view && IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
        Vector2 delta = GetMouseDelta();
        engine->camera.target_goal_x -= (double)delta.x / engine->camera.zoom_goal;
        engine->camera.target_goal_y -= (double)delta.y / engine->camera.zoom_goal;
    }

    if (mouse_in_view && !mouse_in_minimap && IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 delta = GetMouseDelta();
        if (!CheckCollisionPointRec(mouse, (Rectangle){0, 0, (float)GetScreenWidth(), engine->camera.top_bar_height})) {
            engine->camera.target_goal_x -= (double)delta.x / engine->camera.zoom_goal;
            engine->camera.target_goal_y -= (double)delta.y / engine->camera.zoom_goal;
        }
    }
}

void blueprint_engine_update(BlueprintEngine *engine, float dt) {
    blueprint_engine_set_viewport(engine, GetScreenWidth(), GetScreenHeight());
    handle_camera_input(engine);

    double smoothing = clampd(dt * 10.0, 0.0, 1.0);
    engine->camera.target_x += (engine->camera.target_goal_x - engine->camera.target_x) * smoothing;
    engine->camera.target_y += (engine->camera.target_goal_y - engine->camera.target_y) * smoothing;
    engine->camera.zoom += (engine->camera.zoom_goal - engine->camera.zoom) * smoothing;

    if (!engine->paused) {
        engine->time_seconds += dt;
        engine->signal_phase += dt * 1.2;
    }
}

void blueprint_draw_world_grid(const BlueprintEngine *engine) {
    double view_world_w = engine->camera.viewport_width / engine->camera.zoom;
    double target_spacing = view_world_w / 10.0;
    double exponent = floor(log10(target_spacing > 1e-12 ? target_spacing : 1.0));
    double base = pow(10.0, exponent);
    double spacing = base;
    double multiples[3] = {1.0, 2.0, 5.0};
    for (int i = 0; i < 3; ++i) {
        if (base * multiples[i] >= target_spacing) {
            spacing = base * multiples[i];
            break;
        }
    }

    double minor = spacing / 5.0;
    DVec2 top_left;
    DVec2 bottom_right;
    get_world_view_bounds(engine, &top_left, &bottom_right);

    double start_x_minor = floor(top_left.x / minor) * minor;
    double start_y_minor = floor(top_left.y / minor) * minor;
    double start_x_major = floor(top_left.x / spacing) * spacing;
    double start_y_major = floor(top_left.y / spacing) * spacing;

    Color minor_color = (Color){24, 33, 44, 255};
    Color major_color = (Color){42, 58, 76, 255};
    Color axis_color = (Color){100, 140, 200, 255};

    for (double x = start_x_minor; x <= bottom_right.x; x += minor) {
        Color c = fabs(fmod(x, spacing)) < minor * 0.25 ? major_color : minor_color;
        if (fabs(x) < minor * 0.25) c = axis_color;
        Vector2 a = blueprint_world_to_screen(engine, dvec2(x, top_left.y));
        Vector2 b = blueprint_world_to_screen(engine, dvec2(x, bottom_right.y));
        DrawLineEx(a, b, c.r == axis_color.r ? 1.5f : 1.0f, c);
    }
    for (double y = start_y_minor; y <= bottom_right.y; y += minor) {
        Color c = fabs(fmod(y, spacing)) < minor * 0.25 ? major_color : minor_color;
        if (fabs(y) < minor * 0.25) c = axis_color;
        Vector2 a = blueprint_world_to_screen(engine, dvec2(top_left.x, y));
        Vector2 b = blueprint_world_to_screen(engine, dvec2(bottom_right.x, y));
        DrawLineEx(a, b, c.r == axis_color.r ? 1.5f : 1.0f, c);
    }

    for (double x = start_x_major; x <= bottom_right.x; x += spacing) {
        Vector2 p = blueprint_world_to_screen(engine, dvec2(x, top_left.y));
        char label[64];
        snprintf(label, sizeof(label), "%.0f", x);
        DrawText(label, (int)p.x + 4, (int)engine->camera.top_bar_height + 6, 12, (Color){135, 155, 185, 255});
    }
    for (double y = start_y_major; y <= bottom_right.y; y += spacing) {
        Vector2 p = blueprint_world_to_screen(engine, dvec2(top_left.x, y));
        char label[64];
        snprintf(label, sizeof(label), "%.0f", y);
        DrawText(label, 6, (int)p.y + 4, 12, (Color){135, 155, 185, 255});
    }
}

void blueprint_draw_arrow(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color) {
    Vector2 a = blueprint_world_to_screen(engine, from);
    Vector2 b = blueprint_world_to_screen(engine, to);
    DrawLineEx(a, b, thickness, color);

    DVec2 dir = dvec2_normalize(dvec2_sub(to, from));
    DVec2 perp = dvec2_perp(dir);
    double head = 18.0 / engine->camera.zoom;
    DVec2 left = dvec2_sub(to, dvec2_add(dvec2_scale(dir, head), dvec2_scale(perp, head * 0.45)));
    DVec2 right = dvec2_sub(to, dvec2_sub(dvec2_scale(dir, head), dvec2_scale(perp, head * 0.45)));
    Vector2 l = blueprint_world_to_screen(engine, left);
    Vector2 r = blueprint_world_to_screen(engine, right);
    DrawTriangle(b, l, r, color);
}

void blueprint_draw_signal_arrow(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color, double phase_offset) {
    blueprint_draw_arrow(engine, from, to, thickness, color);

    DVec2 dir = dvec2_sub(to, from);
    double len = dvec2_length(dir);
    if (len <= 1e-9) return;
    DVec2 unit = dvec2_scale(dir, 1.0 / len);
    double phase = fmod(engine->signal_phase + phase_offset, 1.0);
    double pulse_pos = len * phase;
    DVec2 pulse = dvec2_add(from, dvec2_scale(unit, pulse_pos));
    Vector2 p = blueprint_world_to_screen(engine, pulse);
    DrawCircleV(p, clampf(3.0f + thickness * 0.7f, 3.0f, 8.0f), color_lerp(color, WHITE, 0.35f));
}

void blueprint_draw_directed_edge(const BlueprintEngine *engine, DVec2 from, DVec2 to, float thickness, Color color, bool curved) {
    if (!curved) {
        blueprint_draw_arrow(engine, from, to, thickness, color);
        return;
    }

    DVec2 delta = dvec2_sub(to, from);
    DVec2 mid = dvec2_scale(dvec2_add(from, to), 0.5);
    DVec2 control = dvec2_add(mid, dvec2_scale(dvec2_perp(delta), 0.18));
    Vector2 from_screen = blueprint_world_to_screen(engine, from);
    Vector2 to_screen = blueprint_world_to_screen(engine, to);
    float dx = to_screen.x - from_screen.x;
    float dy = to_screen.y - from_screen.y;
    float screen_length = sqrtf(dx * dx + dy * dy);
    int segments = (int)clampf(screen_length / 18.0f, 6.0f, 24.0f);
    DVec2 last = from;
    for (int i = 1; i <= segments; ++i) {
        double t = (double)i / (double)segments;
        double u = 1.0 - t;
        DVec2 p = dvec2(
            u * u * from.x + 2.0 * u * t * control.x + t * t * to.x,
            u * u * from.y + 2.0 * u * t * control.y + t * t * to.y
        );
        Vector2 a = blueprint_world_to_screen(engine, last);
        Vector2 b = blueprint_world_to_screen(engine, p);
        DrawLineEx(a, b, thickness, color);
        last = p;
    }
    DVec2 tangent = dvec2_sub(to, control);
    DVec2 back = dvec2_sub(to, dvec2_scale(dvec2_normalize(tangent), 20.0 / engine->camera.zoom));
    blueprint_draw_arrow(engine, back, to, thickness, color);
}

void blueprint_draw_matrix_grid(const BlueprintEngine *engine, DVec2 origin, int rows, int cols, double cell_size, Color line_color, float thickness) {
    for (int r = 0; r <= rows; ++r) {
        DVec2 a = dvec2(origin.x, origin.y + r * cell_size);
        DVec2 b = dvec2(origin.x + cols * cell_size, origin.y + r * cell_size);
        Vector2 sa = blueprint_world_to_screen(engine, a);
        Vector2 sb = blueprint_world_to_screen(engine, b);
        DrawLineEx(sa, sb, thickness, line_color);
    }
    for (int c = 0; c <= cols; ++c) {
        DVec2 a = dvec2(origin.x + c * cell_size, origin.y);
        DVec2 b = dvec2(origin.x + c * cell_size, origin.y + rows * cell_size);
        Vector2 sa = blueprint_world_to_screen(engine, a);
        Vector2 sb = blueprint_world_to_screen(engine, b);
        DrawLineEx(sa, sb, thickness, line_color);
    }
}

void blueprint_draw_heatmap(const BlueprintEngine *engine, DVec2 origin, const HeatmapData *heatmap) {
    const Color cool = (Color){28, 98, 176, 255};
    const Color warm = (Color){242, 115, 48, 255};
    const Color neutral = (Color){34, 38, 50, 255};

    for (int r = 0; r < heatmap->rows; ++r) {
        for (int c = 0; c < heatmap->cols; ++c) {
            float v = heatmap->values[r * heatmap->cols + c];
            float t = (v + 1.0f) * 0.5f;
            Color mixed = color_lerp(cool, warm, t);
            Color fill = color_lerp(neutral, mixed, 0.85f);
            DVec2 cell = dvec2(origin.x + c * heatmap->cell_size, origin.y + r * heatmap->cell_size);
            Vector2 min = blueprint_world_to_screen(engine, cell);
            Vector2 max = blueprint_world_to_screen(engine, dvec2(cell.x + heatmap->cell_size, cell.y + heatmap->cell_size));
            DrawRectangleV(min, (Vector2){max.x - min.x, max.y - min.y}, fill);
        }
    }
    blueprint_draw_matrix_grid(engine, origin, heatmap->rows, heatmap->cols, heatmap->cell_size, (Color){196, 214, 238, 255}, 1.0f);
}

void blueprint_draw_equation_block(const BlueprintEngine *engine, DVec2 origin, const char *title, const char *lines[], int line_count, Color color) {
    Vector2 anchor = blueprint_world_to_screen(engine, origin);
    float scale = clampf((float)(engine->camera.zoom * 0.16), 0.45f, 2.2f);
    int title_size = (int)(18 * scale);
    int line_size = (int)(16 * scale);
    if (title_size < 10) title_size = 10;
    if (line_size < 9) line_size = 9;
    DrawText(title, (int)anchor.x, (int)anchor.y, title_size, color);
    for (int i = 0; i < line_count; ++i) {
        DrawText(lines[i], (int)anchor.x, (int)anchor.y + title_size + 6 + i * (line_size + 4), line_size, color);
    }
}

void blueprint_draw_dense_cluster(const BlueprintEngine *engine, const GraphClusterData *cluster, DVec2 origin, Color node_color, Color edge_color) {
    for (int i = 0; i < cluster->edge_count; ++i) {
        DVec2 from = dvec2_add(origin, cluster->edges[i].from);
        DVec2 to = dvec2_add(origin, cluster->edges[i].to);
        if (!blueprint_world_segment_visible(engine, from, to, 32.0)) {
            continue;
        }
        blueprint_draw_directed_edge(engine, from, to, 1.0f, edge_color, (i % 3) == 0);
    }
    float radius = clampf(blueprint_world_length_to_screen(engine, cluster->radius), 1.0f, 4.0f);
    for (int i = 0; i < cluster->point_count; ++i) {
        DVec2 world = dvec2_add(origin, cluster->points[i]);
        if (!blueprint_world_point_visible(engine, world, cluster->radius * 4.0)) {
            continue;
        }
        Vector2 p = blueprint_world_to_screen(engine, world);
        DrawCircleV(p, radius, node_color);
    }
}

void blueprint_draw_covariance_ellipse(const BlueprintEngine *engine, DVec2 origin, const CovarianceData *cov, Color color) {
    double trace = cov->sigma_xx + cov->sigma_yy;
    double det_term = sqrt(fmax(0.0, trace * trace * 0.25 - (cov->sigma_xx * cov->sigma_yy - cov->sigma_xy * cov->sigma_xy)));
    double lambda1 = trace * 0.5 + det_term;
    double lambda2 = trace * 0.5 - det_term;
    double axis_a = sqrt(fmax(lambda1, 1e-9)) * cov->radius_scale;
    double axis_b = sqrt(fmax(lambda2, 1e-9)) * cov->radius_scale;
    double angle = 0.5 * atan2(2.0 * cov->sigma_xy, cov->sigma_xx - cov->sigma_yy);

    const int segments = 96;
    DVec2 points[96];
    for (int i = 0; i < segments; ++i) {
        double t = ((double)i / (double)segments) * 2.0 * PI;
        double ex = cos(t) * axis_a;
        double ey = sin(t) * axis_b;
        double rx = cos(angle) * ex - sin(angle) * ey;
        double ry = sin(angle) * ex + cos(angle) * ey;
        points[i] = dvec2(origin.x + rx, origin.y + ry);
    }
    draw_world_polyline(engine, points, segments, 2.0f, color, true);
    blueprint_draw_arrow(engine, origin, dvec2(origin.x + cos(angle) * axis_a, origin.y + sin(angle) * axis_a), 2.0f, color_lerp(color, WHITE, 0.2f));
    blueprint_draw_arrow(engine, origin, dvec2(origin.x - sin(angle) * axis_b, origin.y + cos(angle) * axis_b), 2.0f, color_lerp(color, WHITE, 0.4f));
}

static void draw_node_layer(BlueprintEngine *engine, BlueprintLayer layer) {
    Camera2D snapshot = blueprint_camera_snapshot(engine);
    for (size_t i = 0; i < engine->count; ++i) {
        BlueprintNode *node = &engine->nodes[i];
        if (!node->visible || node->layer != layer || node->draw == NULL || node->page != engine->active_page) {
            continue;
        }
        if (!blueprint_world_rect_visible(engine, node->bounds_min, node->bounds_max, 80.0 / engine->camera.zoom)) {
            continue;
        }
        g_active_node = node;
        node->draw(snapshot);
    }
    g_active_node = NULL;
}

void blueprint_engine_draw(BlueprintEngine *engine) {
    g_engine = engine;

    BeginDrawing();
    ClearBackground((Color){8, 11, 16, 255});
    draw_top_bar(engine);

    if (engine->active_page == 0 || engine->active_page == 1) {
        blueprint_draw_world_grid(engine);
        draw_node_layer(engine, BLUEPRINT_LAYER_GEOMETRY);
        draw_node_layer(engine, BLUEPRINT_LAYER_SIGNALS);
        draw_node_layer(engine, BLUEPRINT_LAYER_MATH);
        draw_minimap(engine);
        draw_debug_overlay(engine);
    } else {
        DrawText("Page 2", 20, (int)engine->camera.top_bar_height + 20, 24, (Color){220, 225, 235, 255});
        DrawText("reserved for additional technical canvases", 20, (int)engine->camera.top_bar_height + 52, 18, (Color){130, 145, 165, 255});
    }

    EndDrawing();
    g_engine = NULL;
}
