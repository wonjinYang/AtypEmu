FROM python:3.13.7-slim-bookworm@sha256:781449467ffb6f04218f09b1ecdcdc7d22b289ee5da9ec498b024e24ad7a6db7

ENV OPENMM_DEFAULT_PLATFORM=Reference \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1

COPY wheelhouse/ /tmp/wheelhouse/
RUN test "$(find /tmp/wheelhouse -mindepth 1 -maxdepth 1 -printf x | wc -c)" = 2 && \
    test -f /tmp/wheelhouse/numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl && \
    test -f /tmp/wheelhouse/openmm-8.6.0-cp313-cp313-manylinux_2_34_x86_64.whl && \
    cd /tmp/wheelhouse && \
    printf '%s  %s\n%s  %s\n' \
      5b83648633d46f77039c29078751f80da65aa64d5622a3cd62aaef9d835b6c93 \
      numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
      1b0f39a812452fd9eb1faf28ed4e9d832aef44fc2072f4442a81a4ce7a6f56bf \
      openmm-8.6.0-cp313-cp313-manylinux_2_34_x86_64.whl | sha256sum -c - && \
    python -m pip install --no-cache-dir --no-deps \
      numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
      openmm-8.6.0-cp313-cp313-manylinux_2_34_x86_64.whl && \
    rm -rf /tmp/wheelhouse
