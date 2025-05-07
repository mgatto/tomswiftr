FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
LABEL org.opencontainers.image.authors="gatto_omar@hotmail.com"
#gatto_omar@hotmail.com

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN uv run -- spacy download en_core_web_sm

#TODO don't I have to create my own pykernel?

# MAYBE   VOLUME [“/notebooks”]

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

EXPOSE 8888

# TODO uv run --with jupyter jupyter notebook

# CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]
# ENTRYPOINT ["python3", "-m"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
# TODO add this above --PasswordIdentityProvider.hashed_password=''


# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container
# CMD ["fastapi", "dev", "--host", "0.0.0.0", "src/uv_docker_example"]
# TODO this should be the jupyter command
# RUN uv run some_script.py