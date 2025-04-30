"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html.
"""

import polars as pl
from kedro.config import OmegaConfigLoader
from omegaconf.resolvers import oc

from kedro_mlops.library.mlflow.hooks import MlflowHook

# Instantiate and list your project hooks here
HOOKS = (MlflowHook(),)

# List the installed plugins for which to disable auto-registry
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Define the configuration folder. Defaults to `conf`
CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "config_patterns": {
        "parameters": [
            "parameters*",
            "parameters*/**",
            "*/parameters*",
            "**/parameters*",
        ],
        "mlflow": ["mlflow*", "mlflow*/**", "**/mlflow*"],
        # Note: for globals, we use the default e.g. "globals": ["globals.yml"],
    },
    "merge_strategy": {
        "parameters": "soft",
    },
    "custom_resolvers": {
        "env": oc.env,
        "polars": lambda x: getattr(pl, x),
    },
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
