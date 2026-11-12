# shellcheck shell=bash
# Machine-local overrides for the model/slot paths. The file is gitignored:
# on a machine that needs one, .env is a symlink to
# ~/.config/environment.d/model-router.conf (the same file the user manager
# reads for the service). Values already set in the environment win, so an
# explicit export on the command line still works.
_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
_model_router_env_file="${MODEL_ROUTER_ENV_FILE:-$_repo_root/.env}"
if [[ -r "$_model_router_env_file" ]]; then
  while IFS='=' read -r _k _v; do
    [[ -z "$_k" || "$_k" == \#* ]] && continue
    [[ -n "${!_k:-}" ]] && continue
    export "$_k=$_v"
  done <"$_model_router_env_file"
fi
unset _k _v _repo_root _model_router_env_file
