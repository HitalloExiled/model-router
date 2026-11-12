# shellcheck shell=bash
# Machine-local overrides, kept outside the repo (the systemd unit reads the
# same file through EnvironmentFile). Values already set in the environment
# win over the file, so an explicit export on the command line still works.
_model_router_env_file="${MODEL_ROUTER_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/model-router/env}"
if [[ -r "$_model_router_env_file" ]]; then
  while IFS='=' read -r _k _v; do
    [[ -z "$_k" || "$_k" == \#* ]] && continue
    [[ -n "${!_k:-}" ]] && continue
    export "$_k=$_v"
  done <"$_model_router_env_file"
fi
unset _k _v _model_router_env_file
