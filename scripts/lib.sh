# shellcheck shell=bash
# Shared helpers for the workspace wrapper scripts in scripts/.
# Source this file, then call the helpers; do not execute it directly.
#
#   source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
#   enter_ros_workspace
#   exec ros2 launch eeg_bci_pipeline <launch> "$@"

# Resolve paths from this library's own location (scripts/ sits one level below the repo root).
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ros_distro="${ROS_DISTRO:-jazzy}"
ros_setup="/opt/ros/${ros_distro}/setup.bash"
workspace_setup="${repo_root}/install/setup.bash"

require_ros() {
  if [ ! -f "$ros_setup" ]; then
    echo "ROS setup not found at $ros_setup" >&2
    exit 1
  fi
}

require_ros_and_workspace() {
  require_ros
  if [ ! -f "$workspace_setup" ]; then
    echo "Workspace is not built. Run: scripts/build" >&2
    exit 1
  fi
}

# ROS setup scripts are not nounset-clean, so toggle -u off around sourcing.
source_ros() {
  set +u
  # shellcheck source=/dev/null
  source "$ros_setup"
  set -u
}

source_ros_workspace() {
  set +u
  # shellcheck source=/dev/null
  source "$ros_setup"
  # shellcheck source=/dev/null
  source "$workspace_setup"
  set -u
}

# Source ROS and the workspace when each is present, without failing if absent.
source_ros_workspace_optional() {
  set +u
  if [ -f "$ros_setup" ]; then
    # shellcheck source=/dev/null
    source "$ros_setup"
  fi
  if [ -f "$workspace_setup" ]; then
    # shellcheck source=/dev/null
    source "$workspace_setup"
  fi
  set -u
}

# Verify prerequisites, source the environment, and cd to the repo root.
enter_ros_workspace() {
  require_ros_and_workspace
  source_ros_workspace
  cd "$repo_root" || exit 1
}

# Put a pipx bin dir on PATH so user-installed dev tools (ruff, basedpyright) resolve.
prepend_pipx_bin() {
  local pipx_bin="${PIPX_BIN_DIR:-$HOME/.local/bin}"
  if [ -d "$pipx_bin" ]; then
    PATH="$pipx_bin:$PATH"
  fi
}

# Resolve how to invoke ruff into the RUFF array, or exit 127 with install guidance.
# shellcheck disable=SC2034  # RUFF is consumed by the scripts that source this lib.
ensure_ruff() {
  if command -v ruff >/dev/null 2>&1; then
    RUFF=(ruff)
  elif python3 -m ruff --version >/dev/null 2>&1; then
    RUFF=(python3 -m ruff)
  else
    cat >&2 <<'EOF'
ruff is not installed.
Run:
  scripts/setup-dev-tools
EOF
    exit 127
  fi
}
