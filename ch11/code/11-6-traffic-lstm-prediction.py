#!/usr/bin/env python3
"""
11.6 실습: LSTM을 활용한 도시 교통량 예측

이 스크립트는 LSTM 모델로 교통량을 예측하고,
MC Dropout을 통해 불확실성을 정량화한다.

목표:
- 과거 24시간 데이터 → 미래 1시간 교통량 예측
- 예측 신뢰구간 제공 (MC Dropout)

특성:
- volume: 교통량 (대/시간)
- speed: 평균 속도 (km/h)
- hour: 시간 (0-23)
- day_of_week: 요일 (0-6)
- is_weekend: 주말 여부 (0/1)

실행:
  python practice/chapter11/code/11-6-traffic-lstm-prediction.py

출력:
  - practice/chapter11/output/traffic_prediction_result.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 재현성을 위한 시드 고정
np.random.seed(42)
torch.manual_seed(42)


class TrafficLSTM(nn.Module):
    """교통량 예측 LSTM 모델 (MC Dropout 지원)"""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 마지막 시점 출력
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        pred = self.fc(last_out)
        return pred


def generate_traffic_data(n_days: int = 60, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    현실적인 교통 패턴을 가진 시뮬레이션 데이터 생성

    Args:
        n_days: 생성할 일수
        seed: 랜덤 시드

    Returns:
        volume: 시간당 교통량 (n_days * 24,)
        speed: 시간당 평균 속도 (n_days * 24,)
    """
    np.random.seed(seed)
    n_hours = n_days * 24
    hours = np.arange(n_hours)

    # 시간대별 기본 패턴 (출퇴근 피크)
    hour_of_day = hours % 24
    base_pattern = np.zeros(n_hours)

    # 오전 피크 (7-9시)
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8) / 1.5) ** 2)
    # 오후 피크 (17-19시)
    evening_peak = np.exp(-0.5 * ((hour_of_day - 18) / 1.5) ** 2)
    # 점심 시간 (12-13시)
    lunch_peak = 0.3 * np.exp(-0.5 * ((hour_of_day - 12.5) / 1) ** 2)

    base_pattern = 400 + 600 * (morning_peak + evening_peak + lunch_peak)

    # 주말 감소
    day_of_week = (hours // 24) % 7
    is_weekend = (day_of_week >= 5).astype(float)
    weekend_factor = 1 - 0.4 * is_weekend

    # 노이즈 추가
    noise = np.random.normal(0, 50, n_hours)

    volume = base_pattern * weekend_factor + noise
    volume = np.clip(volume, 50, 1500)

    # 속도: 교통량과 역상관
    base_speed = 80 - 0.03 * volume + np.random.normal(0, 5, n_hours)
    speed = np.clip(base_speed, 20, 100)

    return volume.astype(np.float32), speed.astype(np.float32)


def create_features(volume: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """교통 데이터에서 특성 추출"""
    n = len(volume)
    hours = np.arange(n)

    hour_of_day = (hours % 24).astype(np.float32) / 23  # 정규화
    day_of_week = ((hours // 24) % 7).astype(np.float32) / 6  # 정규화
    is_weekend = ((hours // 24) % 7 >= 5).astype(np.float32)

    features = np.stack([volume, speed, hour_of_day, day_of_week, is_weekend], axis=1)
    return features


def create_sequences(
    data: np.ndarray, target_col: int = 0, seq_len: int = 24, pred_len: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """시퀀스 데이터 생성"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len, target_col])
    return np.array(X), np.array(y)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.001,
    verbose: bool = True,
) -> list[float]:
    """모델 학습"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)
            print(f"Epoch [{epoch+1:3d}/{epochs}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return train_losses


def predict_with_uncertainty(
    model: nn.Module, X: torch.Tensor, n_samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """MC Dropout을 이용한 불확실성 추정"""
    model.train()  # Dropout 활성화
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X)
            predictions.append(pred.numpy())

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scaler: MinMaxScaler) -> dict[str, float]:
    """예측 성능 평가"""
    # 역정규화
    y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-6))) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    scaler: MinMaxScaler,
    save_path: Path,
) -> None:
    """예측 결과 시각화"""
    # 역정규화
    y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # 표준편차도 역정규화 (스케일만 적용)
    scale = scaler.scale_[0]
    y_std_orig = y_std.flatten() / scale

    # 최근 100시간만 표시
    n_show = min(100, len(y_true_orig))

    plt.figure(figsize=(14, 6))

    x = np.arange(n_show)
    plt.plot(x, y_true_orig[:n_show], "b-", label="Actual", linewidth=1.5)
    plt.plot(x, y_pred_orig[:n_show], "r-", label="Predicted", linewidth=1.5)

    # 95% 신뢰구간 (±1.96 * std)
    lower = y_pred_orig[:n_show] - 1.96 * y_std_orig[:n_show]
    upper = y_pred_orig[:n_show] + 1.96 * y_std_orig[:n_show]
    plt.fill_between(x, lower, upper, color="red", alpha=0.2, label="95% CI")

    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Traffic Volume (vehicles/hour)", fontsize=12)
    plt.title("Traffic Volume Prediction with Uncertainty (LSTM + MC Dropout)", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[저장] {save_path}")


def main() -> None:
    # 경로 설정
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("11.6 실습: LSTM을 활용한 도시 교통량 예측")
    print("=" * 60)

    # 1. 데이터 생성
    print("\n[1] 데이터 생성 중...")
    volume, speed = generate_traffic_data(n_days=60)
    features = create_features(volume, speed)
    print(f"    총 데이터: {len(volume)} 시간 ({len(volume)//24}일)")

    # 2. 정규화
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # 3. 시퀀스 생성
    seq_len = 24
    pred_len = 1
    X, y = create_sequences(features_scaled, target_col=0, seq_len=seq_len, pred_len=pred_len)
    print(f"    시퀀스: {X.shape[0]}개 (입력: {seq_len}시간 → 출력: {pred_len}시간)")

    # 4. 학습/검증/테스트 분할 (7:1:2)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"    학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")

    # Tensor 변환
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)

    # 5. 모델 학습
    print("\n[2] 모델 학습 중...")
    model = TrafficLSTM(input_dim=5, hidden_dim=64, num_layers=2, dropout=0.2)
    train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, epochs=50, lr=0.001)

    # 6. 예측 및 불확실성 추정
    print("\n[3] 예측 및 불확실성 추정 중...")
    mean_pred, std_pred = predict_with_uncertainty(model, X_test_t, n_samples=100)

    # 7. 평가
    print("\n[4] 성능 평가")
    # volume 열만 역정규화하기 위한 scaler 생성
    volume_scaler = MinMaxScaler()
    volume_scaler.fit(volume.reshape(-1, 1))

    metrics = evaluate(y_test, mean_pred, volume_scaler)
    print(f"    MAE:  {metrics['mae']:.2f} 대/시간")
    print(f"    RMSE: {metrics['rmse']:.2f} 대/시간")
    print(f"    MAPE: {metrics['mape']:.2f}%")

    # 불확실성 통계
    std_orig = std_pred.flatten() / volume_scaler.scale_[0]
    print(f"\n    불확실성 (MC Dropout 100회):")
    print(f"    평균 표준편차: ±{std_orig.mean():.2f} 대/시간")
    print(f"    95% 신뢰구간: ±{1.96 * std_orig.mean():.2f} 대/시간")

    # 8. 시각화
    print("\n[5] 결과 시각화...")
    plot_results(
        y_test,
        mean_pred,
        std_pred,
        volume_scaler,
        output_dir / "traffic_prediction_result.png",
    )

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
