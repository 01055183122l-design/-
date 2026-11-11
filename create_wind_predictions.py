import csv
import math
import sys
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


INPUT_FILE = "wind_dataset.csv"
OUTPUT_FILE = "predictions.csv"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURE_COLUMNS = [
    "wind_speed_ms",
    "wind_speed_1h_ago_ms",
    "wind_speed_2h_ago_ms",
    "temperature_C",
    "humidity_pct",
]
TARGET_COLUMN = "actual_wind_speed_1h_ahead_ms"
OUTPUT_COLUMNS = [
    "prediction_time",
    "target_time",
    "actual_wind_speed_ms",
    "current_wind_persistence_ms",
    "linear_regression_prediction_ms",
    "random_forest_prediction_ms",
]
TRAIN_START = datetime(2023, 1, 1)
TEST_START = datetime(2024, 12, 1)
TEST_END = datetime(2025, 1, 1)


def solve_linear_system(matrix, vector):
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        pivot = augmented[pivot_row][column]
        if abs(pivot) < 1e-12:
            raise ValueError("Training data does not support a unique linear regression fit")
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        pivot = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= pivot

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]

    return [augmented[row][size] for row in range(size)]


def fit_linear_regression(features, targets):
    row_count = len(features)
    feature_count = len(FEATURE_COLUMNS)
    means = [sum(row[column] for row in features) / row_count for column in range(feature_count)]
    scales = []
    for column, mean in enumerate(means):
        variance = sum((row[column] - mean) ** 2 for row in features) / row_count
        scale = variance**0.5
        if scale == 0:
            raise ValueError(f"Training feature has no variation: {FEATURE_COLUMNS[column]}")
        scales.append(scale)

    target_mean = sum(targets) / row_count
    gram = [[0.0] * feature_count for _ in range(feature_count)]
    covariance = [0.0] * feature_count
    for row, target in zip(features, targets):
        normalized = [
            (row[column] - means[column]) / scales[column]
            for column in range(feature_count)
        ]
        centered_target = target - target_mean
        for left in range(feature_count):
            covariance[left] += normalized[left] * centered_target
            for right in range(feature_count):
                gram[left][right] += normalized[left] * normalized[right]

    coefficients = solve_linear_system(gram, covariance)
    return means, scales, target_mean, coefficients


def predict(features, model):
    means, scales, target_mean, coefficients = model
    prediction = target_mean + sum(
        coefficient * (value - mean) / scale
        for value, mean, scale, coefficient in zip(features, means, scales, coefficients)
    )
    return max(0.0, prediction)


def main():
    folder = Path(__file__).resolve().parent
    input_path = folder / INPUT_FILE
    output_path = folder / OUTPUT_FILE
    required_columns = {
        "prediction_time",
        "target_time",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    }

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{input_path.name} is missing required columns")
        rows = list(reader)

    for row in rows:
        row["_target_datetime"] = datetime.strptime(row["target_time"], TIME_FORMAT)
        row["_prediction_datetime"] = datetime.strptime(row["prediction_time"], TIME_FORMAT)
    rows.sort(key=lambda row: (row["_target_datetime"], row["_prediction_datetime"]))

    training_rows = [
        row for row in rows if TRAIN_START <= row["_target_datetime"] < TEST_START
    ]
    evaluation_rows = [
        row for row in rows if TEST_START <= row["_target_datetime"] < TEST_END
    ]
    if not training_rows or not evaluation_rows:
        raise ValueError(
            f"Expected non-empty training and December 2024 evaluation sets; "
            f"got {len(training_rows)} and {len(evaluation_rows)} rows"
        )

    def numeric_values(row):
        values = [float(row[column]) for column in FEATURE_COLUMNS]
        target = float(row[TARGET_COLUMN])
        if not all(math.isfinite(value) for value in values + [target]):
            raise ValueError("Dataset contains a non-finite numeric value")
        return values, target

    training_data = [numeric_values(row) for row in training_rows]
    model = fit_linear_regression(
        [features for features, _ in training_data],
        [target for _, target in training_data],
    )
    random_forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    random_forest.fit(
        [features for features, _ in training_data],
        [target for _, target in training_data],
    )

    predictions = []
    absolute_errors_persistence = []
    absolute_errors_regression = []
    absolute_errors_random_forest = []
    for row in evaluation_rows:
        features, actual = numeric_values(row)
        persistence_prediction = features[0]
        regression_prediction = predict(features, model)
        random_forest_prediction = float(random_forest.predict([features])[0])
        predictions.append(
            [
                row["prediction_time"],
                row["target_time"],
                actual,
                persistence_prediction,
                regression_prediction,
                random_forest_prediction,
            ]
        )
        absolute_errors_persistence.append(abs(actual - persistence_prediction))
        absolute_errors_regression.append(abs(actual - regression_prediction))
        absolute_errors_random_forest.append(abs(actual - random_forest_prediction))

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(predictions)

    persistence_mae = sum(absolute_errors_persistence) / len(predictions)
    regression_mae = sum(absolute_errors_regression) / len(predictions)
    random_forest_mae = sum(absolute_errors_random_forest) / len(predictions)
    print(f"학습 행수: {len(training_rows)}")
    print(f"평가 행수: {len(evaluation_rows)}")
    print(f"현재 풍속 유지 MAE: {persistence_mae:.6f} m/s")
    print(f"선형 회귀 MAE: {regression_mae:.6f} m/s")
    print(f"랜덤 포레스트 MAE: {random_forest_mae:.6f} m/s")
    print(
        f"선형 회귀 MAE 변화 (기준 대비): "
        f"{regression_mae - persistence_mae:+.6f} m/s"
    )
    print(
        f"랜덤 포레스트 MAE 변화 (기준 대비): "
        f"{random_forest_mae - persistence_mae:+.6f} m/s"
    )
    print(
        f"랜덤 포레스트 MAE 변화 (선형 회귀 대비): "
        f"{random_forest_mae - regression_mae:+.6f} m/s"
    )
    print("첫 평가 결과:")
    writer = csv.writer(sys.stdout)
    writer.writerow(OUTPUT_COLUMNS)
    writer.writerow(predictions[0])


if __name__ == "__main__":
    main()
