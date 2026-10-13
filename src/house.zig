//! Task 3: House Price Prediction, from scratch, zero dependencies.
//! Uses the California housing CSV
//!   * data exploration: per-feature summary table + price histogram
//!   * median imputation, one-hot ocean proximity, engineered ratio features
//!   * standardisation (training-set statistics only)
//!   * Ridge regression via normal equations (Gaussian elimination, pivoting),
//!     regularization strength picked by 5-fold cross-validation
//!   * plain linear regression (no penalty) by full-batch gradient descent
//!   * MSE / RMSE / MAE / R^2, baseline, coefficients, residual histogram
//!
//!   zig build run-house -- data/housing.csv

const std = @import("std");
const Allocator = std.mem.Allocator;

const feature_count = 16;
const feature_names = [feature_count][]const u8{
    "longitude",     "latitude",      "housing_age",  "total_rooms",
    "total_bedrms",  "population",    "households",   "median_income",
    "<1H_ocean",     "inland",        "island",       "near_bay",
    "near_ocean",    "rooms_per_hh",  "bedrm_ratio",  "pop_per_hh",
};

// Column positions inside one feature row.
const total_rooms_column = 3;
const total_bedrooms_column = 4;
const population_column = 5;
const households_column = 6;
const first_ocean_proximity_column = 8;
const rooms_per_household_column = 13;
const bedroom_ratio_column = 14;
const people_per_household_column = 15;

const csv_column_count = 10;
const csv_price_column = 8;
const csv_ocean_proximity_column = 9;

const price_histogram_bucket_width = 50_000.0;
const residual_histogram_bucket_width = 40_000.0;
const histogram_bucket_count = 10;
const histogram_bar_width: usize = 40;

fn parseNumberOrNaN(text: []const u8) f64 {
    return std.fmt.parseFloat(f64, std.mem.trim(u8, text, " ")) catch std.math.nan(f64);
}

fn oceanProximityCategoryIndex(category_name: []const u8) ?usize {
    const category_names = [_][]const u8{ "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN" };
    for (category_names, 0..) |known_category_name, category_index| {
        if (std.mem.eql(u8, known_category_name, category_name)) return category_index;
    }
    return null;
}

/// Reads the CSV into a flat row-major feature matrix plus a price vector.
/// Rows with a missing price or an unknown ocean-proximity label are skipped.
fn parseHousingCsv(
    allocator: Allocator,
    csv_text: []const u8,
    feature_matrix: *std.ArrayList(f64),
    house_prices: *std.ArrayList(f64),
) !void {
    var csv_lines = std.mem.splitScalar(u8, csv_text, '\n');
    _ = csv_lines.next(); // header row
    while (csv_lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r");
        if (line.len == 0) continue;

        var fields: [csv_column_count][]const u8 = undefined;
        var field_count: usize = 0;
        var field_iterator = std.mem.splitScalar(u8, line, ',');
        while (field_iterator.next()) |field| : (field_count += 1) {
            if (field_count < csv_column_count) fields[field_count] = field;
        }
        if (field_count != csv_column_count) continue;

        const price = parseNumberOrNaN(fields[csv_price_column]);
        if (std.math.isNan(price)) continue;
        const ocean_category_index = oceanProximityCategoryIndex(std.mem.trim(u8, fields[csv_ocean_proximity_column], " ")) orelse continue;

        var feature_row = [_]f64{0} ** feature_count;
        for (0..8) |column_index| feature_row[column_index] = parseNumberOrNaN(fields[column_index]);
        feature_row[first_ocean_proximity_column + ocean_category_index] = 1;
        feature_row[rooms_per_household_column] = feature_row[total_rooms_column] / feature_row[households_column];
        feature_row[bedroom_ratio_column] = feature_row[total_bedrooms_column] / feature_row[total_rooms_column]; // NaN if bedrooms missing
        feature_row[people_per_household_column] = feature_row[population_column] / feature_row[households_column];

        try feature_matrix.appendSlice(allocator, &feature_row);
        try house_prices.append(allocator, price);
    }
}

/// Solves matrix * solution = right_hand_side in place with Gaussian elimination
/// and partial pivoting. A (numerically) singular pivot yields a weight of 0.
fn solveLinearSystem(matrix: []f64, right_hand_side: []f64, solution: []f64) void {
    for (0..feature_count) |pivot_column| {
        var pivot_row = pivot_column;
        for (pivot_column + 1..feature_count) |candidate_row| {
            if (@abs(matrix[candidate_row * feature_count + pivot_column]) > @abs(matrix[pivot_row * feature_count + pivot_column])) {
                pivot_row = candidate_row;
            }
        }
        if (pivot_row != pivot_column) {
            for (0..feature_count) |column_index| {
                std.mem.swap(f64, &matrix[pivot_column * feature_count + column_index], &matrix[pivot_row * feature_count + column_index]);
            }
            std.mem.swap(f64, &right_hand_side[pivot_column], &right_hand_side[pivot_row]);
        }

        const pivot_value = matrix[pivot_column * feature_count + pivot_column];
        if (@abs(pivot_value) < 1e-12) continue;
        for (pivot_column + 1..feature_count) |row_index| {
            const elimination_factor = matrix[row_index * feature_count + pivot_column] / pivot_value;
            if (elimination_factor == 0) continue;
            for (pivot_column..feature_count) |column_index| {
                matrix[row_index * feature_count + column_index] -= elimination_factor * matrix[pivot_column * feature_count + column_index];
            }
            right_hand_side[row_index] -= elimination_factor * right_hand_side[pivot_column];
        }
    }

    var row_index: usize = feature_count;
    while (row_index > 0) {
        row_index -= 1;
        var remaining_value = right_hand_side[row_index];
        for (row_index + 1..feature_count) |column_index| {
            remaining_value -= matrix[row_index * feature_count + column_index] * solution[column_index];
        }
        const diagonal_value = matrix[row_index * feature_count + row_index];
        solution[row_index] = if (@abs(diagonal_value) < 1e-12) 0 else remaining_value / diagonal_value;
    }
}

fn fitRidgeRegression(
    allocator: Allocator,
    feature_matrix: []const f64,
    house_prices: []const f64,
    training_rows: []const usize,
    mean_price: f64,
    regularization_strength: f64,
    weights: []f64,
) !void {
    const gram_matrix = try allocator.alloc(f64, feature_count * feature_count);
    @memset(gram_matrix, 0);
    const feature_price_products = try allocator.alloc(f64, feature_count);
    @memset(feature_price_products, 0);

    for (training_rows) |row_index| {
        const feature_row = feature_matrix[row_index * feature_count ..][0..feature_count];
        const centered_price = house_prices[row_index] - mean_price;
        for (0..feature_count) |first_feature| {
            feature_price_products[first_feature] += feature_row[first_feature] * centered_price;
            for (0..feature_count) |second_feature| {
                gram_matrix[first_feature * feature_count + second_feature] += feature_row[first_feature] * feature_row[second_feature];
            }
        }
    }
    // The intercept is the centred mean price, so only the weights are penalised.
    for (0..feature_count) |feature_index| {
        gram_matrix[feature_index * feature_count + feature_index] += regularization_strength;
    }
    solveLinearSystem(gram_matrix, feature_price_products, weights);
}

fn predictPrice(feature_matrix: []const f64, row_index: usize, mean_price: f64, weights: []const f64) f64 {
    var predicted_price = mean_price;
    for (0..feature_count) |feature_index| {
        predicted_price += feature_matrix[row_index * feature_count + feature_index] * weights[feature_index];
    }
    return predicted_price;
}

fn computeRootMeanSquaredError(
    feature_matrix: []const f64,
    house_prices: []const f64,
    rows: []const usize,
    mean_price: f64,
    weights: []const f64,
) f64 {
    var sum_of_squared_errors: f64 = 0;
    for (rows) |row_index| {
        const prediction_error = predictPrice(feature_matrix, row_index, mean_price, weights) - house_prices[row_index];
        sum_of_squared_errors += prediction_error * prediction_error;
    }
    return @sqrt(sum_of_squared_errors / @as(f64, @floatFromInt(rows.len)));
}

/// Plain linear regression (no penalty), full-batch gradient descent on the MSE.
///
/// The gradient (2/n) * (gram_matrix * weights - feature_price_products) depends
/// on the data only through gram_matrix = X^T X and feature_price_products =
/// X^T (price - mean), so both are computed ONCE in a single pass over the data.
/// Every epoch then costs feature_count^2 operations instead of rows * features:
/// the same maths and the same result, roughly a thousand times faster.
/// Features are standardised, so a single learning rate suits every weight.
fn fitLinearRegressionByGradientDescent(
    stdout: *std.Io.Writer,
    allocator: Allocator,
    feature_matrix: []const f64,
    house_prices: []const f64,
    training_rows: []const usize,
    mean_price: f64,
    learning_rate: f64,
    epoch_count: usize,
    weights: []f64,
) !void {
    const gram_matrix = try allocator.alloc(f64, feature_count * feature_count);
    @memset(gram_matrix, 0);
    const feature_price_products = try allocator.alloc(f64, feature_count);
    @memset(feature_price_products, 0);
    var centered_price_sum_of_squares: f64 = 0;

    for (training_rows) |row_index| {
        const feature_row = feature_matrix[row_index * feature_count ..][0..feature_count];
        const centered_price = house_prices[row_index] - mean_price;
        centered_price_sum_of_squares += centered_price * centered_price;
        for (0..feature_count) |first_feature| {
            feature_price_products[first_feature] += feature_row[first_feature] * centered_price;
            for (0..feature_count) |second_feature| {
                gram_matrix[first_feature * feature_count + second_feature] += feature_row[first_feature] * feature_row[second_feature];
            }
        }
    }

    @memset(weights, 0);
    const gradient = try allocator.alloc(f64, feature_count);
    const training_row_count: f64 = @floatFromInt(training_rows.len);

    for (0..epoch_count) |epoch_index| {
        for (0..feature_count) |first_feature| {
            var gradient_component = -feature_price_products[first_feature];
            for (0..feature_count) |second_feature| {
                gradient_component += gram_matrix[first_feature * feature_count + second_feature] * weights[second_feature];
            }
            gradient[first_feature] = gradient_component;
        }

        const is_progress_epoch = epoch_index == 0 or epoch_index == 9 or epoch_index == 99 or
            epoch_index == 999 or epoch_index + 1 == epoch_count;
        if (is_progress_epoch) {
            // sum of squared errors = y'y - 2 w'b + w'Aw, evaluated without touching the rows.
            var weights_dot_products: f64 = 0;
            var weights_quadratic_form: f64 = 0;
            for (0..feature_count) |feature_index| {
                weights_dot_products += weights[feature_index] * feature_price_products[feature_index];
                weights_quadratic_form += weights[feature_index] * (gradient[feature_index] + feature_price_products[feature_index]);
            }
            const sum_of_squared_errors = @max(0.0, centered_price_sum_of_squares - 2 * weights_dot_products + weights_quadratic_form);
            try stdout.print("  epoch {d:>5}  train RMSE ${d:.0}\n", .{ epoch_index + 1, @sqrt(sum_of_squared_errors / training_row_count) });
        }

        for (0..feature_count) |feature_index| {
            weights[feature_index] -= learning_rate * 2.0 * gradient[feature_index] / training_row_count;
        }
    }
}

fn printTestMetrics(
    stdout: *std.Io.Writer,
    model_name: []const u8,
    feature_matrix: []const f64,
    house_prices: []const f64,
    rows: []const usize,
    mean_price: f64,
    weights: []const f64,
) !void {
    const row_count: f64 = @floatFromInt(rows.len);
    var mean_actual_price: f64 = 0;
    for (rows) |row_index| mean_actual_price += house_prices[row_index];
    mean_actual_price /= row_count;

    var sum_of_squared_errors: f64 = 0;
    var sum_of_absolute_errors: f64 = 0;
    var total_sum_of_squares: f64 = 0;
    for (rows) |row_index| {
        const prediction_error = predictPrice(feature_matrix, row_index, mean_price, weights) - house_prices[row_index];
        sum_of_squared_errors += prediction_error * prediction_error;
        sum_of_absolute_errors += @abs(prediction_error);
        const deviation_from_mean = house_prices[row_index] - mean_actual_price;
        total_sum_of_squares += deviation_from_mean * deviation_from_mean;
    }
    try stdout.print(
        "  {s:<24} MSE {e:.3}  RMSE ${d:.0}  MAE ${d:.0}  R^2 {d:.4}\n",
        .{ model_name, sum_of_squared_errors / row_count, @sqrt(sum_of_squared_errors / row_count), sum_of_absolute_errors / row_count, 1.0 - sum_of_squared_errors / total_sum_of_squares },
    );
}

fn printHistogramBar(stdout: *std.Io.Writer, bucket_count: usize, largest_bucket_count: usize) !void {
    const bar_length = bucket_count * histogram_bar_width / largest_bucket_count;
    for (0..bar_length) |_| try stdout.print("#", .{});
}

/// Summary statistics of the raw data (before any imputation) and a price histogram.
fn exploreData(
    stdout: *std.Io.Writer,
    allocator: Allocator,
    feature_matrix: []const f64,
    house_prices: []const f64,
) !void {
    const house_count = house_prices.len;
    try stdout.print("\n{s:<14} {s:>10} {s:>10} {s:>10} {s:>10} {s:>8}\n", .{ "feature", "min", "mean", "median", "max", "missing" });
    for (0..feature_count) |feature_index| {
        var present_values: std.ArrayList(f64) = .empty;
        var missing_count: usize = 0;
        for (0..house_count) |row_index| {
            const value = feature_matrix[row_index * feature_count + feature_index];
            if (std.math.isNan(value)) {
                missing_count += 1;
            } else {
                try present_values.append(allocator, value);
            }
        }
        std.mem.sort(f64, present_values.items, {}, std.sort.asc(f64));
        var sum_of_values: f64 = 0;
        for (present_values.items) |value| sum_of_values += value;
        const present_count = present_values.items.len;
        try stdout.print(
            "{s:<14} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>8}\n",
            .{ feature_names[feature_index], present_values.items[0], sum_of_values / @as(f64, @floatFromInt(present_count)), present_values.items[present_count / 2], present_values.items[present_count - 1], missing_count },
        );
    }

    var houses_per_price_bucket = [_]usize{0} ** histogram_bucket_count;
    for (house_prices) |price| {
        const bucket_index: usize = @min(histogram_bucket_count - 1, @as(usize, @intFromFloat(@max(0, price) / price_histogram_bucket_width)));
        houses_per_price_bucket[bucket_index] += 1;
    }
    var largest_bucket_count: usize = 1;
    for (houses_per_price_bucket) |bucket_count| largest_bucket_count = @max(largest_bucket_count, bucket_count);
    try stdout.print("\nprice distribution ($50k buckets, last bucket is 450k+):\n", .{});
    for (houses_per_price_bucket, 0..) |bucket_count, bucket_index| {
        try stdout.print("  {d:>4}k ", .{bucket_index * 50});
        try printHistogramBar(stdout, bucket_count, largest_bucket_count);
        try stdout.print(" {d}\n", .{bucket_count});
    }
}

/// Orders feature indices by descending absolute weight.
const DescendingByAbsoluteWeight = struct {
    weights: []const f64,

    pub fn lessThan(self: DescendingByAbsoluteWeight, first_index: usize, second_index: usize) bool {
        return @abs(self.weights[first_index]) > @abs(self.weights[second_index]);
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();
    const command_line_arguments = try init.minimal.args.toSlice(allocator);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;
    defer stdout.flush() catch {};

    if (command_line_arguments.len < 2) {
        try stdout.print("usage: house <housing.csv>\n", .{});
        return;
    }

    var feature_matrix_list: std.ArrayList(f64) = .empty;
    var house_prices_list: std.ArrayList(f64) = .empty;
    const csv_text = try std.Io.Dir.cwd().readFileAlloc(io, command_line_arguments[1], allocator, .limited(256 * 1024 * 1024));
    try parseHousingCsv(allocator, csv_text, &feature_matrix_list, &house_prices_list);
    const feature_matrix = feature_matrix_list.items;
    const house_prices = house_prices_list.items;
    const house_count = house_prices.len;
    if (house_count < 50) {
        try stdout.print("only {d} usable rows; is this the California housing CSV?\n", .{house_count});
        return;
    }

    try exploreData(stdout, allocator, feature_matrix, house_prices);

    // ---- 80/20 shuffled train/test split -----------------------------------
    var random_generator = std.Random.DefaultPrng.init(2026);
    const shuffled_row_indices = try allocator.alloc(usize, house_count);
    for (shuffled_row_indices, 0..) |*slot, position| slot.* = position;
    random_generator.random().shuffle(usize, shuffled_row_indices);
    const test_rows = shuffled_row_indices[0 .. house_count / 5];
    const training_rows = shuffled_row_indices[house_count / 5 ..];
    const training_row_count: f64 = @floatFromInt(training_rows.len);

    // ---- impute with training medians, then standardise with training mean/std
    var missing_cell_count: usize = 0;
    for (feature_matrix) |value| missing_cell_count += @intFromBool(std.math.isNan(value));
    for (0..feature_count) |feature_index| {
        var training_values: std.ArrayList(f64) = .empty;
        for (training_rows) |row_index| {
            const value = feature_matrix[row_index * feature_count + feature_index];
            if (!std.math.isNan(value)) try training_values.append(allocator, value);
        }
        std.mem.sort(f64, training_values.items, {}, std.sort.asc(f64));
        const training_median = training_values.items[training_values.items.len / 2];
        for (0..house_count) |row_index| {
            if (std.math.isNan(feature_matrix[row_index * feature_count + feature_index])) {
                feature_matrix[row_index * feature_count + feature_index] = training_median;
            }
        }

        var training_mean: f64 = 0;
        for (training_rows) |row_index| training_mean += feature_matrix[row_index * feature_count + feature_index];
        training_mean /= training_row_count;
        var sum_of_squared_deviations: f64 = 0;
        for (training_rows) |row_index| {
            const deviation = feature_matrix[row_index * feature_count + feature_index] - training_mean;
            sum_of_squared_deviations += deviation * deviation;
        }
        var training_standard_deviation = @sqrt(sum_of_squared_deviations / training_row_count);
        if (training_standard_deviation < 1e-12) training_standard_deviation = 1;
        for (0..house_count) |row_index| {
            const cell_index = row_index * feature_count + feature_index;
            feature_matrix[cell_index] = (feature_matrix[cell_index] - training_mean) / training_standard_deviation;
        }
    }
    var mean_training_price: f64 = 0;
    for (training_rows) |row_index| mean_training_price += house_prices[row_index];
    mean_training_price /= training_row_count;

    try stdout.print(
        "\n{d} houses | train {d} / test {d} | {d} missing cells imputed | {d} features\n",
        .{ house_count, training_rows.len, test_rows.len, missing_cell_count, feature_count },
    );

    // ---- pick the regularization strength by 5-fold CV on the training set -
    const candidate_regularization_strengths = [_]f64{ 0.01, 0.1, 1, 10, 100, 1000 };
    const cross_validation_fold_count = 5;
    const weights = try allocator.alloc(f64, feature_count);
    var best_regularization_strength: f64 = candidate_regularization_strengths[0];
    var lowest_cross_validation_error: f64 = std.math.inf(f64);
    try stdout.print("\n5-fold CV (RMSE, $):\n", .{});
    for (candidate_regularization_strengths) |regularization_strength| {
        var summed_validation_error: f64 = 0;
        for (0..cross_validation_fold_count) |fold_index| {
            var fold_training_rows: std.ArrayList(usize) = .empty;
            var fold_validation_rows: std.ArrayList(usize) = .empty;
            for (training_rows, 0..) |row_index, position| {
                if (position % cross_validation_fold_count == fold_index) {
                    try fold_validation_rows.append(allocator, row_index);
                } else {
                    try fold_training_rows.append(allocator, row_index);
                }
            }
            try fitRidgeRegression(allocator, feature_matrix, house_prices, fold_training_rows.items, mean_training_price, regularization_strength, weights);
            summed_validation_error += computeRootMeanSquaredError(feature_matrix, house_prices, fold_validation_rows.items, mean_training_price, weights);
        }
        const average_validation_error = summed_validation_error / cross_validation_fold_count;
        try stdout.print("  lambda {d:>8.2}  ->  {d:.0}\n", .{ regularization_strength, average_validation_error });
        if (average_validation_error < lowest_cross_validation_error) {
            lowest_cross_validation_error = average_validation_error;
            best_regularization_strength = regularization_strength;
        }
    }

    // ---- final Ridge fit and test-set metrics ------------------------------
    try fitRidgeRegression(allocator, feature_matrix, house_prices, training_rows, mean_training_price, best_regularization_strength, weights);
    const test_row_count: f64 = @floatFromInt(test_rows.len);
    var mean_test_price: f64 = 0;
    for (test_rows) |row_index| mean_test_price += house_prices[row_index];
    mean_test_price /= test_row_count;

    var sum_of_squared_errors: f64 = 0;
    var sum_of_absolute_errors: f64 = 0;
    var total_sum_of_squares: f64 = 0;
    var houses_per_residual_bucket = [_]usize{0} ** histogram_bucket_count;
    for (test_rows) |row_index| {
        const prediction_error = predictPrice(feature_matrix, row_index, mean_training_price, weights) - house_prices[row_index];
        sum_of_squared_errors += prediction_error * prediction_error;
        sum_of_absolute_errors += @abs(prediction_error);
        const deviation_from_mean = house_prices[row_index] - mean_test_price;
        total_sum_of_squares += deviation_from_mean * deviation_from_mean;

        const unclamped_bucket: i64 = @intFromFloat(@floor((prediction_error + 200_000.0) / residual_histogram_bucket_width));
        houses_per_residual_bucket[@intCast(std.math.clamp(unclamped_bucket, 0, histogram_bucket_count - 1))] += 1;
    }
    try stdout.print("\nbest lambda {d}\n", .{best_regularization_strength});
    try stdout.print(
        "TEST  MSE {e:.4}  RMSE ${d:.0}  MAE ${d:.0}  R^2 {d:.4}\n",
        .{ sum_of_squared_errors / test_row_count, @sqrt(sum_of_squared_errors / test_row_count), sum_of_absolute_errors / test_row_count, 1.0 - sum_of_squared_errors / total_sum_of_squares },
    );
    try stdout.print("baseline (always predict mean): RMSE ${d:.0}\n", .{@sqrt(total_sum_of_squares / test_row_count)});

    // ---- plain linear regression (no penalty) by gradient descent ----------
    try stdout.print("\n== Linear regression by gradient descent (no penalty) ==\n", .{});
    const gradient_descent_weights = try allocator.alloc(f64, feature_count);
    try fitLinearRegressionByGradientDescent(stdout, allocator, feature_matrix, house_prices, training_rows, mean_training_price, 0.05, 5000, gradient_descent_weights);
    try stdout.print("\ntest-set comparison:\n", .{});
    try printTestMetrics(stdout, "Linear (grad. descent)", feature_matrix, house_prices, test_rows, mean_training_price, gradient_descent_weights);
    try printTestMetrics(stdout, "Ridge (closed form)", feature_matrix, house_prices, test_rows, mean_training_price, weights);

    // ---- what drives price? standardised coefficients ----------------------
    try stdout.print("\ncoefficients ($ per +1 std-dev):\n", .{});
    const feature_indices_by_importance = try allocator.alloc(usize, feature_count);
    for (feature_indices_by_importance, 0..) |*slot, position| slot.* = position;
    std.mem.sort(usize, feature_indices_by_importance, DescendingByAbsoluteWeight{ .weights = weights }, DescendingByAbsoluteWeight.lessThan);
    for (feature_indices_by_importance[0..6]) |feature_index| {
        try stdout.print("  {s:<14} {d:>10.0}\n", .{ feature_names[feature_index], weights[feature_index] });
    }

    // ---- residual histogram (predicted - actual, $40k buckets) -------------
    try stdout.print("\nresiduals (pred - actual):\n", .{});
    var largest_residual_bucket: usize = 1;
    for (houses_per_residual_bucket) |bucket_count| largest_residual_bucket = @max(largest_residual_bucket, bucket_count);
    for (houses_per_residual_bucket, 0..) |bucket_count, bucket_index| {
        const bucket_lower_bound_in_thousands: i64 = -200 + @as(i64, @intCast(bucket_index)) * 40;
        try stdout.print("  {d:>5}k ", .{bucket_lower_bound_in_thousands});
        try printHistogramBar(stdout, bucket_count, largest_residual_bucket);
        try stdout.print(" {d}\n", .{bucket_count});
    }
}
