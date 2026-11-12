# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.patch.worker import patch_uva


@pytest.mark.parametrize(
    "installed,expected",
    [
        ("3.2.1", False),
        ("3.2.2", False),
        ("3.2.2+dev20260729205041", False),
        ("3.2.1.dev1", False),
        ("3.2.2.post1", False),
        ("3.2.0", True),
        ("3.2.3+dev1", True),
    ],
)
def test_uva_version_guard_includes_development_wheels(monkeypatch, installed, expected):
    monkeypatch.setattr(patch_uva, "version", lambda name: installed)
    assert patch_uva.check_triton_ascend_version_valid() is expected
