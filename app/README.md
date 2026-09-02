# Eureka V10 Integration App

Primary demo: `python -m voice_app.web`. The FastAPI gateway owns conversation
state, cancellation and two WebSocket planes: replayable control events and
framed PCM. The browser uses AudioWorklet for bounded playout plus live PCM
capture for push-to-talk and VAD/barge-in.
