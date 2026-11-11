import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path


INPUT_FILE = "asos_108_2023_2024.csv"
OUTPUT_FILE = "wind_dataset.csv"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

OUTPUT_COLUMNS = [
    "prediction_time",
    "target_time",
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_ago_ms",
    "wind_speed_2h_ago_ms",
    "actual_wind_speed_1h_ahead_ms",
]


def main():
    folder = Path(__file__).resolve().parent
    input_path = folder / INPUT_FILE
    output_path = folder / OUTPUT_FILE

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {
            "datetime",
            "temperature_C",
            "humidity_pct",
            "wind_speed_ms",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{input_path.name} is missing required columns")

        observations = {}
        for row in reader:
            timestamp = datetime.strptime(row["datetime"], TIME_FORMAT)
            if timestamp.minute or timestamp.second or timestamp.microsecond:
                raise ValueError(f"Timestamp is not on an hour boundary: {row['datetime']}")
            if timestamp in observations:
                raise ValueError(f"Duplicate timestamp: {row['datetime']}")
            observations[timestamp] = row

    if not observations:
        raise ValueError(f"{input_path.name} contains no observations")

    first_time = min(observations)
    last_time = max(observations)
    aligned_times = []
    timestamp = first_time
    while timestamp <= last_time:
        aligned_times.append(timestamp)
        timestamp += timedelta(hours=1)

    def value_at(timestamp, column):
        row = observations.get(timestamp)
        value = row.get(column, "") if row else ""
        return value.strip() if value is not None else ""

    output_rows = []
    for index in range(2, len(aligned_times) - 1):
        prediction_time = aligned_times[index]
        previous_time = prediction_time - timedelta(hours=1)
        two_hours_ago = prediction_time - timedelta(hours=2)
        target_time = prediction_time + timedelta(hours=1)
        values = [
            value_at(prediction_time, "wind_speed_ms"),
            value_at(prediction_time, "temperature_C"),
            value_at(prediction_time, "humidity_pct"),
            value_at(previous_time, "wind_speed_ms"),
            value_at(two_hours_ago, "wind_speed_ms"),
            value_at(target_time, "wind_speed_ms"),
        ]
        if any(value == "" for value in values):
            continue

        output_rows.append(
            [
                prediction_time.strftime(TIME_FORMAT),
                target_time.strftime(TIME_FORMAT),
                *values[:3],
                *values[3:5],
                values[5],
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(output_rows)

    print(f"시간 간격을 맞춘 뒤의 행수: {len(aligned_times)}")
    print(f"저장된 행수: {len(output_rows)}")
    print("첫 4행:")
    writer = csv.writer(sys.stdout)
    writer.writerow(OUTPUT_COLUMNS)
    writer.writerows(output_rows[:4])


if __name__ == "__main__":
    main()
