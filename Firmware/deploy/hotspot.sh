#!/usr/bin/env bash
set -euo pipefail

# Create a local Wi-Fi access point using NetworkManager. Run as root after
# NetworkManager is confirmed to manage the Pi's wireless adapter.
: "${TOPGUN_WIFI_IFACE:?Set TOPGUN_WIFI_IFACE to the Pi wireless interface}"
: "${TOPGUN_HOTSPOT_SSID:?Set TOPGUN_HOTSPOT_SSID}"
: "${TOPGUN_HOTSPOT_PASSWORD:?Set TOPGUN_HOTSPOT_PASSWORD (8+ characters)}"
if [[ ${#TOPGUN_HOTSPOT_PASSWORD} -lt 8 ]]; then
  echo "TOPGUN_HOTSPOT_PASSWORD must be at least 8 characters" >&2
  exit 2
fi
if ! command -v nmcli >/dev/null 2>&1; then
  echo "nmcli not found; install/enable NetworkManager on this Pi OS image first" >&2
  exit 2
fi

connection_name=topgun-hotspot
if nmcli -t -f NAME connection show | grep -Fxq "$connection_name"; then
  nmcli connection delete "$connection_name"
fi
nmcli connection add type wifi ifname "$TOPGUN_WIFI_IFACE" \
  con-name "$connection_name" autoconnect yes ssid "$TOPGUN_HOTSPOT_SSID" \
  802-11-wireless.mode ap \
  ipv4.method shared ipv6.method disabled \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$TOPGUN_HOTSPOT_PASSWORD"
nmcli connection up "$connection_name"
echo "Hotspot configured: $TOPGUN_HOTSPOT_SSID"
