//! Task 2: Spam Mail Detector, from scratch, zero dependencies.
//!   * tokenizer + stopword filter     * vocabulary built on TRAIN data only
//!   * Multinomial Naive Bayes         * Logistic Regression on TF-IDF (SGD)
//!   * confusion matrix, precision, recall, F1, ROC-AUC
//!   * "most spammy words" report      * classify your own messages
//!
//!   zig build run-spam -- data/sms.tsv "WINNER!! claim your free prize now"
const std = @import("std");
const Allocator = std.mem.Allocator;

const stop_words = [_][]const u8{
    "the",   "and",  "for",  "you",   "are",   "was",   "with",  "this",
    "that",  "have", "has",  "from",  "but",   "not",   "can",   "its",
    "our",   "out",  "all",  "just",  "then",  "than",  "them",  "they",
    "their", "there", "here", "what", "when",  "who",   "how",   "been",
    "were",  "would", "could", "did",  "does",  "into",  "too",   "she",
    "her",   "him",  "his",  "off",   "any",   "one",   "me",    "my",
    "we",    "us",   "is",   "it",    "in",    "on",    "at",    "to",
    "of",    "or",   "be",   "as",    "by",    "if",    "so",    "do",
    "an",    "am",   "up",   "no",    "he",    "ll",    "ve",    "re",
};

const ham_class_index = 0;
const spam_class_index = 1;

const SparseEntry = struct { feature_index: u32, value: f64 };
const LabeledMessage = struct { is_spam: bool, tokens: []const []const u8 };
const ScoredExample = struct { score: f64, is_spam: bool };

fn isStopWord(word: []const u8) bool {
    for (stop_words) |stop_word| {
        if (std.mem.eql(u8, stop_word, word)) return true;
    }
    return false;
}

fn wordIsAlphabeticallyBefore(_: void, first_word: []const u8, second_word: []const u8) bool {
    return std.mem.lessThan(u8, first_word, second_word);
}

fn scoreIsLower(_: void, first_example: ScoredExample, second_example: ScoredExample) bool {
    return first_example.score < second_example.score;
}

fn sigmoid(input_value: f64) f64 {
    return 1.0 / (1.0 + @exp(-input_value));
}

/// Lowercases the text and splits it into alphanumeric words of 2+ letters,
/// dropping stop words.
fn tokenize(allocator: Allocator, text: []const u8, tokens: *std.ArrayList([]const u8)) !void {
    var cursor: usize = 0;
    while (cursor < text.len) {
        while (cursor < text.len and !std.ascii.isAlphanumeric(text[cursor])) : (cursor += 1) {}
        const word_start = cursor;
        while (cursor < text.len and std.ascii.isAlphanumeric(text[cursor])) : (cursor += 1) {}
        const word_length = cursor - word_start;
        if (word_length < 2) continue;

        const lowercase_word = try allocator.alloc(u8, word_length);
        for (text[word_start..cursor], lowercase_word) |original_character, *lowercase_character| {
            lowercase_character.* = std.ascii.toLower(original_character);
        }
        if (isStopWord(lowercase_word)) continue;
        try tokens.append(allocator, lowercase_word);
    }
}

/// Bag-of-words counts as a sparse vector sorted by feature index.
/// Words that are not in the vocabulary are dropped.
fn countWordOccurrences(
    allocator: Allocator,
    tokens: []const []const u8,
    vocabulary: *const std.StringHashMap(u32),
) ![]SparseEntry {
    var matched_feature_indices: std.ArrayList(u32) = .empty;
    for (tokens) |token| {
        if (vocabulary.get(token)) |feature_index| {
            try matched_feature_indices.append(allocator, feature_index);
        }
    }
    std.mem.sort(u32, matched_feature_indices.items, {}, std.sort.asc(u32));

    var word_counts: std.ArrayList(SparseEntry) = .empty;
    var group_start: usize = 0;
    while (group_start < matched_feature_indices.items.len) {
        var group_end = group_start;
        while (group_end < matched_feature_indices.items.len and
            matched_feature_indices.items[group_end] == matched_feature_indices.items[group_start])
        {
            group_end += 1;
        }
        try word_counts.append(allocator, .{
            .feature_index = matched_feature_indices.items[group_start],
            .value = @floatFromInt(group_end - group_start),
        });
        group_start = group_end;
    }
    return word_counts.items;
}

/// Sublinear term frequency times inverse document frequency, L2-normalised.
fn applyTfIdf(
    allocator: Allocator,
    word_counts: []const SparseEntry,
    inverse_document_frequency: []const f64,
) ![]SparseEntry {
    const weighted_entries = try allocator.alloc(SparseEntry, word_counts.len);
    var squared_length: f64 = 0;
    for (word_counts, weighted_entries) |word_count, *weighted_entry| {
        const tf_idf_value = (1.0 + @log(word_count.value)) * inverse_document_frequency[word_count.feature_index];
        weighted_entry.* = .{ .feature_index = word_count.feature_index, .value = tf_idf_value };
        squared_length += tf_idf_value * tf_idf_value;
    }
    const vector_length = @sqrt(squared_length);
    if (vector_length > 0) {
        for (weighted_entries) |*weighted_entry| weighted_entry.value /= vector_length;
    }
    return weighted_entries;
}

const NaiveBayesModel = struct {
    log_word_probability_by_class: [2][]f64,
    log_class_prior: [2]f64,

    fn logOddsOfSpam(self: NaiveBayesModel, word_counts: []const SparseEntry) f64 {
        var log_score_by_class = self.log_class_prior;
        for (word_counts) |word_count| {
            log_score_by_class[ham_class_index] += word_count.value * self.log_word_probability_by_class[ham_class_index][word_count.feature_index];
            log_score_by_class[spam_class_index] += word_count.value * self.log_word_probability_by_class[spam_class_index][word_count.feature_index];
        }
        return log_score_by_class[spam_class_index] - log_score_by_class[ham_class_index];
    }
};

const LogisticRegressionModel = struct {
    feature_weights: []f64,
    bias: f64,

    fn probabilityOfSpam(self: LogisticRegressionModel, tf_idf_vector: []const SparseEntry) f64 {
        var linear_score = self.bias;
        for (tf_idf_vector) |entry| {
            linear_score += self.feature_weights[entry.feature_index] * entry.value;
        }
        return sigmoid(linear_score);
    }
};

const ConfusionMatrix = struct {
    true_positives: u32 = 0,
    false_positives: u32 = 0,
    true_negatives: u32 = 0,
    false_negatives: u32 = 0,

    fn record(self: *ConfusionMatrix, actually_spam: bool, predicted_spam: bool) void {
        if (actually_spam and predicted_spam) self.true_positives += 1;
        if (!actually_spam and predicted_spam) self.false_positives += 1;
        if (!actually_spam and !predicted_spam) self.true_negatives += 1;
        if (actually_spam and !predicted_spam) self.false_negatives += 1;
    }

    fn print(self: ConfusionMatrix, stdout: *std.Io.Writer, model_name: []const u8) !void {
        const true_positives: f64 = @floatFromInt(self.true_positives);
        const false_positives: f64 = @floatFromInt(self.false_positives);
        const true_negatives: f64 = @floatFromInt(self.true_negatives);
        const false_negatives: f64 = @floatFromInt(self.false_negatives);

        const accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives);
        const precision = if (true_positives + false_positives > 0) true_positives / (true_positives + false_positives) else 0;
        const recall = if (true_positives + false_negatives > 0) true_positives / (true_positives + false_negatives) else 0;
        const f1_score = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0;

        try stdout.print("\n== {s} ==\n", .{model_name});
        try stdout.print("                pred ham  pred spam\n", .{});
        try stdout.print("actual ham   {d:>10}  {d:>9}\n", .{ self.true_negatives, self.false_positives });
        try stdout.print("actual spam  {d:>10}  {d:>9}\n", .{ self.false_negatives, self.true_positives });
        try stdout.print(
            "accuracy {d:.4}  precision {d:.4}  recall {d:.4}  F1 {d:.4}\n",
            .{ accuracy, precision, recall, f1_score },
        );
    }
};

/// ROC-AUC through the rank-sum (Mann-Whitney) formulation; tied scores share their average rank.
fn computeRocAuc(scored_examples: []ScoredExample) f64 {
    std.mem.sort(ScoredExample, scored_examples, {}, scoreIsLower);

    var rank_sum_of_spam: f64 = 0;
    var spam_count: f64 = 0;
    var ham_count: f64 = 0;
    var tie_group_start: usize = 0;
    while (tie_group_start < scored_examples.len) {
        var tie_group_end = tie_group_start;
        while (tie_group_end < scored_examples.len and
            scored_examples[tie_group_end].score == scored_examples[tie_group_start].score)
        {
            tie_group_end += 1;
        }
        const average_rank = @as(f64, @floatFromInt(tie_group_start + 1 + tie_group_end)) / 2.0;
        for (scored_examples[tie_group_start..tie_group_end]) |scored_example| {
            if (scored_example.is_spam) {
                rank_sum_of_spam += average_rank;
                spam_count += 1;
            } else {
                ham_count += 1;
            }
        }
        tie_group_start = tie_group_end;
    }
    return (rank_sum_of_spam - spam_count * (spam_count + 1) / 2) / (spam_count * ham_count);
}

/// Orders feature indices by descending score; used to rank the "most spammy" words.
const DescendingByScore = struct {
    scores: []const f64,

    pub fn lessThan(self: DescendingByScore, first_index: usize, second_index: usize) bool {
        return self.scores[first_index] > self.scores[second_index];
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
        try stdout.print("usage: spam <sms.tsv> [\"message to classify\" ...]\n", .{});
        return;
    }

    // ---- load: one "label<TAB>message" per line ----------------------------
    const dataset_text = try std.Io.Dir.cwd().readFileAlloc(io, command_line_arguments[1], allocator, .limited(256 * 1024 * 1024));
    var labeled_messages: std.ArrayList(LabeledMessage) = .empty;
    var dataset_lines = std.mem.splitScalar(u8, dataset_text, '\n');
    while (dataset_lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, "\r ");
        const tab_position = std.mem.indexOfScalar(u8, line, '\t') orelse continue;
        const label = line[0..tab_position];
        const is_spam = std.mem.eql(u8, label, "spam");
        if (!is_spam and !std.mem.eql(u8, label, "ham")) continue;

        var message_tokens: std.ArrayList([]const u8) = .empty;
        try tokenize(allocator, line[tab_position + 1 ..], &message_tokens);
        try labeled_messages.append(allocator, .{ .is_spam = is_spam, .tokens = message_tokens.items });
    }
    const message_count = labeled_messages.items.len;
    if (message_count < 10) {
        try stdout.print("only {d} usable rows; expected tab-separated label/message lines\n", .{message_count});
        return;
    }

    // ---- 80/20 shuffled train/test split -----------------------------------
    var random_generator = std.Random.DefaultPrng.init(42);
    const random = random_generator.random();
    const shuffled_message_indices = try allocator.alloc(usize, message_count);
    for (shuffled_message_indices, 0..) |*slot, position| slot.* = position;
    random.shuffle(usize, shuffled_message_indices);
    const test_message_count = message_count / 5;
    const test_message_indices = shuffled_message_indices[0..test_message_count];
    const train_message_indices = shuffled_message_indices[test_message_count..];

    // ---- vocabulary from TRAIN data only (word must appear at least twice) -
    var word_frequency = std.StringHashMap(u32).init(allocator);
    for (train_message_indices) |message_index| {
        for (labeled_messages.items[message_index].tokens) |token| {
            const lookup = try word_frequency.getOrPut(token);
            if (!lookup.found_existing) lookup.value_ptr.* = 0;
            lookup.value_ptr.* += 1;
        }
    }
    var vocabulary_words: std.ArrayList([]const u8) = .empty;
    var frequency_iterator = word_frequency.iterator();
    while (frequency_iterator.next()) |frequency_entry| {
        if (frequency_entry.value_ptr.* >= 2) try vocabulary_words.append(allocator, frequency_entry.key_ptr.*);
    }
    // Sorted, so the word -> index mapping is reproducible from run to run.
    std.mem.sort([]const u8, vocabulary_words.items, {}, wordIsAlphabeticallyBefore);
    var vocabulary = std.StringHashMap(u32).init(allocator);
    for (vocabulary_words.items, 0..) |word, word_index| {
        try vocabulary.put(word, @intCast(word_index));
    }
    const vocabulary_size = vocabulary_words.items.len;

    const train_word_counts = try allocator.alloc([]SparseEntry, train_message_indices.len);
    for (train_message_indices, train_word_counts) |message_index, *word_counts| {
        word_counts.* = try countWordOccurrences(allocator, labeled_messages.items[message_index].tokens, &vocabulary);
    }
    const test_word_counts = try allocator.alloc([]SparseEntry, test_message_indices.len);
    for (test_message_indices, test_word_counts) |message_index, *word_counts| {
        word_counts.* = try countWordOccurrences(allocator, labeled_messages.items[message_index].tokens, &vocabulary);
    }

    var spam_message_count: usize = 0;
    for (labeled_messages.items) |labeled_message| spam_message_count += @intFromBool(labeled_message.is_spam);
    try stdout.print(
        "{d} messages ({d} spam, {d} ham) | train {d} / test {d} | vocab {d}\n",
        .{ message_count, spam_message_count, message_count - spam_message_count, train_message_indices.len, test_message_count, vocabulary_size },
    );

    // ---- inverse document frequency ----------------------------------------
    const document_frequency = try allocator.alloc(f64, vocabulary_size);
    @memset(document_frequency, 0);
    for (train_word_counts) |word_counts| {
        for (word_counts) |word_count| document_frequency[word_count.feature_index] += 1;
    }
    const inverse_document_frequency = try allocator.alloc(f64, vocabulary_size);
    const train_message_count_float: f64 = @floatFromInt(train_message_indices.len);
    for (inverse_document_frequency, document_frequency) |*idf_value, word_document_frequency| {
        idf_value.* = @log((1.0 + train_message_count_float) / (1.0 + word_document_frequency)) + 1.0;
    }

    // ---- model 1: Multinomial Naive Bayes with Laplace smoothing -----------
    var word_count_by_class = [2][]f64{ try allocator.alloc(f64, vocabulary_size), try allocator.alloc(f64, vocabulary_size) };
    @memset(word_count_by_class[ham_class_index], 0);
    @memset(word_count_by_class[spam_class_index], 0);
    var total_word_count_by_class = [2]f64{ 0, 0 };
    var message_count_by_class = [2]f64{ 0, 0 };
    for (train_message_indices, train_word_counts) |message_index, word_counts| {
        const class_index: usize = if (labeled_messages.items[message_index].is_spam) spam_class_index else ham_class_index;
        message_count_by_class[class_index] += 1;
        for (word_counts) |word_count| {
            word_count_by_class[class_index][word_count.feature_index] += word_count.value;
            total_word_count_by_class[class_index] += word_count.value;
        }
    }
    const naive_bayes_model: NaiveBayesModel = .{
        .log_word_probability_by_class = .{ try allocator.alloc(f64, vocabulary_size), try allocator.alloc(f64, vocabulary_size) },
        .log_class_prior = .{
            @log(message_count_by_class[ham_class_index] / train_message_count_float),
            @log(message_count_by_class[spam_class_index] / train_message_count_float),
        },
    };
    const vocabulary_size_float: f64 = @floatFromInt(vocabulary_size);
    for ([_]usize{ ham_class_index, spam_class_index }) |class_index| {
        for (0..vocabulary_size) |feature_index| {
            naive_bayes_model.log_word_probability_by_class[class_index][feature_index] =
                @log((word_count_by_class[class_index][feature_index] + 1.0) / (total_word_count_by_class[class_index] + vocabulary_size_float));
        }
    }

    // ---- model 2: Logistic Regression on TF-IDF, SGD with L2 penalty -------
    const train_tf_idf_vectors = try allocator.alloc([]SparseEntry, train_message_indices.len);
    for (train_word_counts, train_tf_idf_vectors) |word_counts, *tf_idf_vector| {
        tf_idf_vector.* = try applyTfIdf(allocator, word_counts, inverse_document_frequency);
    }
    const test_tf_idf_vectors = try allocator.alloc([]SparseEntry, test_message_indices.len);
    for (test_word_counts, test_tf_idf_vectors) |word_counts, *tf_idf_vector| {
        tf_idf_vector.* = try applyTfIdf(allocator, word_counts, inverse_document_frequency);
    }

    var logistic_regression_model: LogisticRegressionModel = .{
        .feature_weights = try allocator.alloc(f64, vocabulary_size),
        .bias = 0,
    };
    @memset(logistic_regression_model.feature_weights, 0);
    const training_order = try allocator.alloc(usize, train_message_indices.len);
    for (training_order, 0..) |*slot, position| slot.* = position;

    const learning_rate = 0.5;
    const l2_penalty_strength = 1e-5;
    const training_epoch_count = 25;
    for (0..training_epoch_count) |_| {
        random.shuffle(usize, training_order);
        for (training_order) |training_position| {
            const target_label: f64 = if (labeled_messages.items[train_message_indices[training_position]].is_spam) 1 else 0;
            const prediction_error = logistic_regression_model.probabilityOfSpam(train_tf_idf_vectors[training_position]) - target_label;
            for (train_tf_idf_vectors[training_position]) |entry| {
                const current_weight = logistic_regression_model.feature_weights[entry.feature_index];
                logistic_regression_model.feature_weights[entry.feature_index] -=
                    learning_rate * (prediction_error * entry.value + l2_penalty_strength * current_weight);
            }
            logistic_regression_model.bias -= learning_rate * prediction_error;
        }
    }

    // ---- evaluate on the held-out test set ---------------------------------
    var naive_bayes_confusion: ConfusionMatrix = .{};
    var logistic_regression_confusion: ConfusionMatrix = .{};
    const naive_bayes_scores = try allocator.alloc(ScoredExample, test_message_count);
    const logistic_regression_scores = try allocator.alloc(ScoredExample, test_message_count);
    for (test_message_indices, test_word_counts, test_tf_idf_vectors, 0..) |message_index, word_counts, tf_idf_vector, test_position| {
        const actually_spam = labeled_messages.items[message_index].is_spam;
        const naive_bayes_log_odds = naive_bayes_model.logOddsOfSpam(word_counts);
        const logistic_regression_probability = logistic_regression_model.probabilityOfSpam(tf_idf_vector);
        naive_bayes_confusion.record(actually_spam, naive_bayes_log_odds > 0);
        logistic_regression_confusion.record(actually_spam, logistic_regression_probability > 0.5);
        naive_bayes_scores[test_position] = .{ .score = naive_bayes_log_odds, .is_spam = actually_spam };
        logistic_regression_scores[test_position] = .{ .score = logistic_regression_probability, .is_spam = actually_spam };
    }
    try naive_bayes_confusion.print(stdout, "Multinomial Naive Bayes");
    try stdout.print("ROC-AUC {d:.4}\n", .{computeRocAuc(naive_bayes_scores)});
    try logistic_regression_confusion.print(stdout, "Logistic Regression (TF-IDF)");
    try stdout.print("ROC-AUC {d:.4}\n", .{computeRocAuc(logistic_regression_scores)});

    // ---- most spam-indicative words ----------------------------------------
    const spam_log_ratio_by_word = try allocator.alloc(f64, vocabulary_size);
    for (spam_log_ratio_by_word, 0..) |*log_ratio, feature_index| {
        const appears_often_enough_in_spam = word_count_by_class[spam_class_index][feature_index] >= 5;
        log_ratio.* = if (appears_often_enough_in_spam)
            naive_bayes_model.log_word_probability_by_class[spam_class_index][feature_index] -
                naive_bayes_model.log_word_probability_by_class[ham_class_index][feature_index]
        else
            -1e9;
    }
    const word_indices_by_spamminess = try allocator.alloc(usize, vocabulary_size);
    for (word_indices_by_spamminess, 0..) |*slot, position| slot.* = position;
    std.mem.sort(usize, word_indices_by_spamminess, DescendingByScore{ .scores = spam_log_ratio_by_word }, DescendingByScore.lessThan);
    try stdout.print("\nmost spammy words (NB log-ratio):", .{});
    for (word_indices_by_spamminess[0..@min(12, vocabulary_size)]) |word_index| {
        try stdout.print(" {s}", .{vocabulary_words.items[word_index]});
    }
    try stdout.print("\n", .{});

    // ---- classify the messages given on the command line -------------------
    for (command_line_arguments[2..]) |user_message| {
        var user_message_tokens: std.ArrayList([]const u8) = .empty;
        try tokenize(allocator, user_message, &user_message_tokens);
        const user_word_counts = try countWordOccurrences(allocator, user_message_tokens.items, &vocabulary);
        const user_tf_idf_vector = try applyTfIdf(allocator, user_word_counts, inverse_document_frequency);
        try stdout.print(
            "\n\"{s}\"\n  NB P(spam) = {d:.3}   LR P(spam) = {d:.3}\n",
            .{ user_message, sigmoid(naive_bayes_model.logOddsOfSpam(user_word_counts)), logistic_regression_model.probabilityOfSpam(user_tf_idf_vector) },
        );
    }
}
