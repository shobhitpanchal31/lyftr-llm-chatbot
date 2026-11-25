#!/bin/sh
set -eu

API_URL="${VITE_API_BASE_URL:-http://localhost:8000}"
ESCAPED_URL=$(printf '%s' "$API_URL" | sed 's/\\/\\\\/g; s/"/\\"/g')

cat <<EOF > /usr/share/nginx/html/config.js
window.__APP_CONFIG__ = {
  API_BASE_URL: "${ESCAPED_URL}"
};
EOF

exec nginx -g 'daemon off;'
