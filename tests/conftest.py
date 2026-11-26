import os

# Required for fail-closed settings. Individual tests may unset this.
os.environ.setdefault("APP_ENV", "local")
