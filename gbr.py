from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)

gbrt_best = GradientBoostingRegressor(
max_depth=2, learning_rate=0.05, n_estimators=500,
n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)

gbrt_best.n_estimators_
