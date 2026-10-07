# Hardware Notes

The current intended bracelet configuration is bipolar acquisition:

```text
In1P/In1N ... In8P/In8N = bipolar electrode pairs
BIAS_EL = common body reference / driven-right-leg electrode
GND = board/USB ground, not a skin electrode
SRB1/SRB2 = off/not used for bipolar mode
```

For 16 channels, the board uses two ADS1299 devices. The original notes describe:

```text
[8ch + (1 ref)] x ADS
Driven right leg circuit
```

That makes `BIAS_EL` the likely reference electrode for a plug-and-play bracelet, not USB/board `GND`.

