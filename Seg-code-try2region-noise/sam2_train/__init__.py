# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
import os

# Only initialize Hydra if not already initialized and not in subprocess
# Check if already initialized by looking for Hydra's config
if not os.environ.get('HYDRA_INITIALIZED'):
    try:
        initialize_config_module("sam2_train", version_base="1.2")
        os.environ['HYDRA_INITIALIZED'] = '1'
    except Exception:
        # May already be initialized in parent process
        pass
