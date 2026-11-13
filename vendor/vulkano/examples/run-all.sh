#!/usr/bin/env sh
set -eu

# This script builds and runs all the examples. It exits as soon as any example exits with a
# nonzero exit code. If the `--headless` argument is given, each example is run for at most 15
# seconds, after which point it moves on to the next example. Otherwise, human input is required to
# close all the windows.

headless=false

for arg; do
  if [ "$arg" = '--headless' ]; then
    headless=true
  else
    echo "$0: unknown argument: $arg"
    exit 1
  fi
done

if [ "$headless" = true ]; then
  sh_cmd='timeout --preserve-status 15s sh'
else
  sh_cmd='sh'
fi

cd "$(dirname $0)"
cargo build --bins
ls -F | grep '/$' | sed 's|/$||' | xargs -E '' -I {} $sh_cmd -c '
  # Continue without printing anything if the example is excluded in the manifest.
  cargo build --bin {} &>/dev/null || exit 0
  # Do not continue if there are errors. An exit status of 255 instructs xargs not to continue.
  cargo run --bin {} || exit 255
'
rm -f pipeline-caching/pipeline_cache.bin
