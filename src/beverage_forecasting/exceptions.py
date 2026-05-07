class ForecastingError(Exception):
    """Base exception for forecasting service errors."""


class ModelUnavailableError(ForecastingError):
    """Raised when an optional model dependency is not installed."""


class NotEnoughHistoryError(ForecastingError):
    """Raised when a state does not have enough history to train safely."""
