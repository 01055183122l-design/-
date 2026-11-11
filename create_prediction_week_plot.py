import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


INPUT_FILE = "predictions.csv"
OUTPUT_FILE = "prediction_week.png"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DECEMBER_START = datetime(2024, 12, 1)
JANUARY_START = datetime(2025, 1, 1)
WEEK_END = datetime(2024, 12, 8)


def main():
    folder = Path(__file__).resolve().parent
    with (folder / INPUT_FILE).open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    december_rows = []
    week_rows = []
    for row in rows:
        target_time = datetime.strptime(row["target_time"], TIME_FORMAT)
        if DECEMBER_START <= target_time < JANUARY_START:
            december_rows.append((target_time, row))
            if DECEMBER_START <= target_time < WEEK_END:
                week_rows.append((target_time, row))

    december_rows.sort(key=lambda item: item[0])
    week_rows.sort(key=lambda item: item[0])
    if len(december_rows) != 744:
        raise ValueError(f"Expected 744 December evaluation rows; found {len(december_rows)}")
    if len(week_rows) != 168:
        raise ValueError(f"Expected 168 evaluation rows for Dec 1-7; found {len(week_rows)}")

    dates = [item[0] for item in week_rows]
    actual = [float(row["actual_wind_speed_ms"]) for _, row in week_rows]
    persistence = [float(row["current_wind_persistence_ms"]) for _, row in week_rows]
    regression = [float(row["linear_regression_prediction_ms"]) for _, row in week_rows]
    random_forest = [
        float(row["random_forest_prediction_ms"]) for _, row in week_rows
    ]

    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, color="black", linewidth=1.5, label="Actual")
    plt.plot(
        dates,
        persistence,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Persistence baseline",
    )
    plt.plot(dates, regression, color="blue", linewidth=1.2, label="Linear regression")
    plt.plot(
        dates,
        random_forest,
        color="green",
        linewidth=1.2,
        label="Random forest",
    )
    plt.xlabel("Target time (KST)")
    plt.ylabel("Wind speed (m/s)")
    plt.title("Wind speed predictions: Dec 1-7, 2024 (KST)")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / OUTPUT_FILE, dpi=160)
    plt.close()

    persistence_mae = sum(
        abs(float(row["actual_wind_speed_ms"]) - float(row["current_wind_persistence_ms"]))
        for _, row in december_rows
    ) / len(december_rows)
    regression_mae = sum(
        abs(float(row["actual_wind_speed_ms"]) - float(row["linear_regression_prediction_ms"]))
        for _, row in december_rows
    ) / len(december_rows)
    random_forest_mae = sum(
        abs(float(row["actual_wind_speed_ms"]) - float(row["random_forest_prediction_ms"]))
        for _, row in december_rows
    ) / len(december_rows)
    print(f"그래프 평가 행수 (12/1~12/7): {len(week_rows)}")
    print(f"MAE 평가 행수 (12/1~12/31): {len(december_rows)}")
    print(f"기준 예측 MAE (12월 전체): {persistence_mae:.6f} m/s")
    print(f"선형 회귀 MAE (12월 전체): {regression_mae:.6f} m/s")
    print(f"랜덤 포레스트 MAE (12월 전체): {random_forest_mae:.6f} m/s")
    print(f"저장 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
