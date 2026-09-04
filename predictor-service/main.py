from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


app = FastAPI()

PROMETHEUS_URL = "http://localhost:9090"

# Шаг получения метрик из Prometheus.
# Одна точка = 15 секунд.
PROMETHEUS_STEP_SECONDS = 15

# Порог CPU (в ядрах) на одну реплику.
CPU_PER_REPLICA = 0.02


class MetricPoint(BaseModel):
    timestamp: float
    value: float


class PredictRequest(BaseModel):
    history: list[MetricPoint]
    horizon_minutes: int = 5


class PredictResponse(BaseModel):
    predicted_value: float
    recommended_replicas: int


class AutoPredictRequest(BaseModel):
    query: str = (
        'sum(rate('
        'container_cpu_usage_seconds_total'
        '{namespace="default",pod=~"demo-app.*"}'
        '[2m]))'
    )
    horizon_minutes: int = 5
    minutes_back: int = 5


def fetch_metrics_from_prometheus(
    query: str,
    minutes_back: int = 5
) -> list[MetricPoint]:
    """
    Получает временной ряд из Prometheus.

    Данные запрашиваются с шагом 15 секунд.
    """

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes_back)

    print(
        f"DEBUG: запрашиваю данные с {start} по {end}"
    )

    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": f"{PROMETHEUS_STEP_SECONDS}s",
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(
            f"Prometheus вернул ошибку: {data}"
        )

    results = data["data"]["result"]

    print(f"DEBUG: получено серий: {len(results)}")

    if not results:
        return []

    print(
        f"DEBUG: точек в первой серии: "
        f"{len(results[0]['values'])}"
    )

    # В текущем PromQL используется sum(...),
    # поэтому ожидается одна агрегированная серия.
    raw_values = results[0]["values"]

    history = []

    for timestamp, value in raw_values:
        try:
            timestamp = float(timestamp)
            value = float(value)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(value):
            continue

        history.append(
            MetricPoint(
                timestamp=timestamp,
                value=value
            )
        )

    return history


def values_to_replicas(predicted: float) -> int:
    """
    Переводит прогнозируемую CPU-нагрузку
    в рекомендуемое количество реплик.
    """

    # Защита от NaN / inf.
    if not np.isfinite(predicted):
        return 2

    # CPU не может быть отрицательным.
    predicted = max(0.0, predicted)

    replicas = int(
        np.ceil(predicted / CPU_PER_REPLICA)
    )

    # Минимум 2 реплики.
    return max(2, replicas)


def forecast_values(
    values: np.ndarray,
    horizon_minutes: int
) -> float:
    """
    Строит прогноз нагрузки.

    horizon_minutes — реальное количество минут,
    на которое нужно сделать прогноз.

    Так как одна точка соответствует 15 секундам,
    количество точек прогноза рассчитывается отдельно.
    """

    if len(values) == 0:
        return 0.0

    # Если горизонт некорректный,
    # используем минимальный горизонт.
    horizon_minutes = max(1, horizon_minutes)

    # Если данных слишком мало,
    # используем последнее значение.
    if len(values) < 10:
        return float(values[-1])

    # Если ряд практически постоянный,
    # Holt-Winters не нужен.
    #
    # Это также предотвращает ситуации,
    # когда SSE становится равным нулю,
    # а statsmodels пытается вычислить log(0)
    # при расчёте AIC/BIC.
    if np.allclose(
        values,
        values[0],
        rtol=1e-5,
        atol=1e-8
    ):
        return float(values[-1])

    # Переводим минуты в количество точек.
    #
    # Например:
    #
    # 5 минут
    # 5 * 60 = 300 секунд
    # 300 / 15 = 20 точек
    #
    forecast_points = max(
        1,
        int(
            np.ceil(
                horizon_minutes
                * 60
                / PROMETHEUS_STEP_SECONDS
            )
        )
    )

    try:
        print("DEBUG: последние 20 значений CPU:")
        print(values[-20:])

        model = ExponentialSmoothing(
            values,
            trend="add",
            damped_trend=True,
            seasonal=None
        ).fit()

        forecast = model.forecast(forecast_points)

        predicted = float(forecast[-1])

    except Exception as exc:
        # Если модель не смогла построиться,
        # используем последнее известное значение.
        print(
            f"WARNING: Holt-Winters не смог построить "
            f"прогноз: {exc}"
        )

        predicted = float(values[-1])

    # Защита от NaN / inf.
    if not np.isfinite(predicted):
        predicted = float(values[-1])

    # CPU не может быть отрицательным.
    return max(0.0, predicted)


@app.post(
    "/predict",
    response_model=PredictResponse
)
def predict(req: PredictRequest):
    """
    Прогноз по истории,
    переданной непосредственно в запросе.
    """

    if not req.history:
        return PredictResponse(
            predicted_value=0.0,
            recommended_replicas=2
        )

    values = np.array(
        [point.value for point in req.history],
        dtype=float
    )

    # Убираем NaN и infinity.
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return PredictResponse(
            predicted_value=0.0,
            recommended_replicas=2
        )

    predicted = forecast_values(
        values,
        req.horizon_minutes
    )

    recommended = values_to_replicas(
        predicted
    )

    return PredictResponse(
        predicted_value=predicted,
        recommended_replicas=recommended
    )


@app.post(
    "/predict/auto",
    response_model=PredictResponse
)
def predict_auto(req: AutoPredictRequest):
    """
    Автоматически получает метрики из Prometheus
    и строит прогноз.
    """

    history = fetch_metrics_from_prometheus(
        req.query,
        req.minutes_back
    )

    if len(history) < 10:
        print(
            "WARNING: недостаточно данных "
            f"для прогноза: {len(history)} точек"
        )

        return PredictResponse(
            predicted_value=0.0,
            recommended_replicas=2
        )

    values = np.array(
        [point.value for point in history],
        dtype=float
    )

    # Убираем некорректные значения.
    values = values[np.isfinite(values)]

    if len(values) < 10:
        return PredictResponse(
            predicted_value=0.0,
            recommended_replicas=2
        )

    predicted = forecast_values(
        values,
        req.horizon_minutes
    )

    recommended = values_to_replicas(
        predicted
    )

    print(
        f"DEBUG: прогноз на "
        f"{req.horizon_minutes} мин: "
        f"{predicted}"
    )

    print(
        f"DEBUG: рекомендуемое количество реплик: "
        f"{recommended}"
    )

    return PredictResponse(
        predicted_value=predicted,
        recommended_replicas=recommended
    )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }