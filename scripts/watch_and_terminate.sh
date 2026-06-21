#!/bin/bash
# Watches RunPod training, downloads checkpoint when done, terminates pod.
# Usage: bash scripts/watch_and_terminate.sh

# ── Config ────────────────────────────────────────────────────────────────────
RUNPOD_HOST="38.147.83.17"
RUNPOD_PORT="41335"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/root/ujamaa-multi-modal"
CHECKPOINT_SIGNAL="$REMOTE_DIR/checkpoints/ujamaa-3b-lora/final/adapter_config.json"
BRIDGE_SIGNAL="$REMOTE_DIR/checkpoints/anthos-bridge/bridge.pt"
LOCAL_SAVE="$HOME/ujamaa-multi-modal/checkpoints"
POLL_INTERVAL=120   # check every 2 minutes
RUNPOD_POD_ID="m01ooy2p6hk4ag"  # from your pod URL

# ── RunPod API key — paste yours here ─────────────────────────────────────────
RUNPOD_API_KEY="rpa_7MHXAIRDXW1UEMY7UX2AYMYBVM68A5TMCT9546JBwkswyn"   # get from runpod.io/console/user/settings

# ── Helpers ───────────────────────────────────────────────────────────────────
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST"
SCP="scp -o StrictHostKeyChecking=no -P $RUNPOD_PORT -i $SSH_KEY"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

check_ssh() {
    $SSH "echo ok" 2>/dev/null | grep -q ok
}

check_done() {
    $SSH "test -f $CHECKPOINT_SIGNAL && echo yes || echo no" 2>/dev/null | grep -q yes
}

get_current_step() {
    $SSH "ls $REMOTE_DIR/checkpoints/ujamaa-3b-lora/ 2>/dev/null | grep checkpoint | tail -1" 2>/dev/null
}

get_loss() {
    $SSH "tail -3 $REMOTE_DIR/logs/ujamaa-3b/*.log 2>/dev/null || echo ''" 2>/dev/null
}

download_checkpoints() {
    log "Downloading LoRA adapter..."
    mkdir -p "$LOCAL_SAVE/ujamaa-3b-lora"
    $SCP -r "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/checkpoints/ujamaa-3b-lora/final" \
        "$LOCAL_SAVE/ujamaa-3b-lora/" && log "LoRA adapter saved to $LOCAL_SAVE/ujamaa-3b-lora/final"

    log "Downloading bridge checkpoint..."
    mkdir -p "$LOCAL_SAVE/anthos-bridge"
    $SCP "$RUNPOD_USER@$RUNPOD_HOST:$BRIDGE_SIGNAL" \
        "$LOCAL_SAVE/anthos-bridge/bridge.pt" && log "Bridge saved to $LOCAL_SAVE/anthos-bridge/bridge.pt"
}

terminate_pod() {
    if [ -z "$RUNPOD_API_KEY" ]; then
        log "No API key set — skipping auto-terminate."
        log "Terminate manually at: https://runpod.io/console/pods"
        return
    fi
    log "Terminating pod $RUNPOD_POD_ID..."
    curl -s -X POST \
        "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) }\"}" \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print('Terminated:', r)" 2>/dev/null \
        || log "Terminate call failed — stop pod manually."
}

notify() {
    # macOS notification
    osascript -e "display notification \"$1\" with title \"Ujamaa Training\"" 2>/dev/null
    log "DONE: $1"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "=== Ujamaa Training Watcher ==="
log "Polling every ${POLL_INTERVAL}s. Press Ctrl+C to stop watching (won't affect training)."
log "Pod: $RUNPOD_HOST:$RUNPOD_PORT"
log ""

FAIL_COUNT=0

while true; do
    if ! check_ssh; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        log "SSH unreachable (attempt $FAIL_COUNT/5)..."
        if [ $FAIL_COUNT -ge 5 ]; then
            log "Pod appears down. Check RunPod dashboard."
            notify "Pod unreachable — check dashboard"
            exit 1
        fi
        sleep $POLL_INTERVAL
        continue
    fi

    FAIL_COUNT=0

    if check_done; then
        log "Training complete! Checkpoint found."
        notify "Training complete — downloading checkpoints"
        download_checkpoints

        # Verify download succeeded
        if [ -f "$LOCAL_SAVE/ujamaa-3b-lora/final/adapter_config.json" ]; then
            log "Checkpoint verified locally."
            terminate_pod
            notify "All done. Pod terminated. Checkpoints at $LOCAL_SAVE"
            exit 0
        else
            log "Download may have failed — NOT terminating pod. Check manually."
            notify "Download failed — check manually before terminating"
            exit 1
        fi
    else
        STEP=$(get_current_step)
        log "Still training... latest checkpoint: ${STEP:-none yet}"
    fi

    sleep $POLL_INTERVAL
done
