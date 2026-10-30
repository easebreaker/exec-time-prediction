"""小规模非负最小二乘。样本量和参数量都很小，不依赖第三方数值库。"""

from __future__ import annotations

import math


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(dot(v, v))


def column_rank(rows: list[list[float]]) -> int:
    """列缩放后的数值秩。成比例的列只计一次。"""
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("设计矩阵列数不一致")
    scales = []
    for column in range(width):
        scales.append(_norm([row[column] for row in rows]) or 1.0)
    columns = [
        [row[column] / scales[column] for row in rows]
        for column in range(width)
    ]
    basis: list[list[float]] = []
    rank = 0
    for column in columns:
        residual = column[:]
        for direction in basis:
            projected = dot(direction, residual)
            residual = [
                value - projected * direction[index]
                for index, value in enumerate(residual)
            ]
        norm = _norm(residual)
        if norm <= 1e-8:
            continue
        basis.append([value / norm for value in residual])
        rank += 1
    return rank


def _lstsq(rows: list[list[float]], targets: list[float]) -> list[float] | None:
    """列满秩时返回最小二乘解。列相关或样本少于参数时返回 None。"""
    n = len(rows)
    if n == 0:
        return None
    p = len(rows[0])
    if n < p or p == 0:
        return None
    cols = [[rows[i][j] for i in range(n)] for j in range(p)]
    q_cols = [[0.0] * n for _ in range(p)]
    r_mat = [[0.0] * p for _ in range(p)]
    for j in range(p):
        v = cols[j][:]
        for i in range(j):
            r_mat[i][j] = dot(q_cols[i], v)
            v = [v[k] - r_mat[i][j] * q_cols[i][k] for k in range(n)]
        r_mat[j][j] = _norm(v)
        col_norm = _norm(cols[j])
        if r_mat[j][j] <= 1e-10 * (1.0 + col_norm):
            return None
        q_cols[j] = [v[k] / r_mat[j][j] for k in range(n)]
    qty = [dot(q_cols[j], targets) for j in range(p)]
    beta = [0.0] * p
    for j in range(p - 1, -1, -1):
        acc = qty[j]
        for k in range(j + 1, p):
            acc -= r_mat[j][k] * beta[k]
        beta[j] = acc / r_mat[j][j]
    return beta


def nnls(rows: list[list[float]], targets: list[float]) -> list[float]:
    """min ||Xθ - y||，θ ≥ 0。参数个数按个位数枚举有效集。

    列先缩放到单位长度，避免序列长度的平方项把其余列淹没。
    """
    if not rows:
        raise ValueError("最小二乘没有样本")
    n = len(rows)
    p = len(rows[0])
    if any(len(row) != p for row in rows):
        raise ValueError("设计矩阵列数不一致")
    if len(targets) != n:
        raise ValueError("目标个数与样本个数不一致")
    scales = []
    for j in range(p):
        scales.append(_norm([rows[i][j] for i in range(n)]) or 1.0)
    scaled = [[rows[i][j] / scales[j] for j in range(p)] for i in range(n)]
    y_scale = max(1.0, _norm(targets))
    neg_tol = 1e-8 * y_scale
    grad_tol = 1e-7 * y_scale
    best: list[float] | None = None
    best_sse = math.inf
    for mask in range(1 << p):
        free = [j for j in range(p) if mask & (1 << j)]
        coef = [0.0] * p
        if free:
            beta = _lstsq([[row[j] for j in free] for row in scaled], targets)
            if beta is None or any(b < -neg_tol for b in beta):
                continue
            for j, b in zip(free, beta):
                coef[j] = 0.0 if b < 0.0 else b
        resid = [
            dot(scaled[i], coef) - targets[i]
            for i in range(n)
        ]
        feasible = True
        for j in range(p):
            if j in free:
                continue
            grad = sum(scaled[i][j] * resid[i] for i in range(n))
            if grad < -grad_tol:
                feasible = False
                break
        if not feasible:
            continue
        sse = dot(resid, resid)
        if sse < best_sse - 1e-18:
            best_sse = sse
            best = coef
    if best is None:
        best = [0.0] * p
    return [best[j] / scales[j] for j in range(p)]


def fit_relative_linear(
    rows: list[list[float]],
    base: list[float],
    measured: list[float],
    n_iter: int = 6,
) -> list[float]:
    """拟合 measured ≈ base + Xθ，θ ≥ 0，目标接近 log 时间上的平方误差。

    列尺度和样本权重分开。每一列先除以自己的峰值，平方项和截距都进到
    一量级，再按当前预测时间的倒数做相对误差。峰值来自原始特征，
    不随 1/预测变化，也不靠极短样本的权重下限去压列范数。
    大 L 的 1/预测很小，但平方项在方程组里的元素是特征/预测：预测跟着
    特征涨时，它趋近常数，不会把系数压成 0。
    无噪声且模型正确时，第一轮就能得到残差为零的参数。
    """
    n = len(rows)
    if n == 0:
        raise ValueError("没有样本")
    p = len(rows[0])
    scales = []
    for j in range(p):
        scales.append(max(abs(rows[i][j]) for i in range(n)) or 1.0)
    scaled_rows = [[rows[i][j] / scales[j] for j in range(p)] for i in range(n)]
    theta = [0.0] * p
    # 权重下限只挡住一条极短样本把 1/时间放到 1e6。列与列差几个数量级
    # 由上面的峰值缩放处理。
    floor = sorted(measured)[n // 2] * 1e-3
    for step in range(n_iter):
        if step == 0:
            preds = measured
        else:
            preds = [base[i] + dot(rows[i], theta) for i in range(n)]
        weighted_rows = []
        weighted_y = []
        for i in range(n):
            weight = 1.0 / max(preds[i], floor, 1e-30)
            weighted_rows.append([weight * scaled_rows[i][j] for j in range(p)])
            weighted_y.append(weight * (measured[i] - base[i]))
        beta = nnls(weighted_rows, weighted_y)
        theta = [beta[j] / scales[j] for j in range(p)]
    return theta
