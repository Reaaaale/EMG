# Serial Protocol

Commands sent by the app:

```text
RDATAC
STOP
READREG1
READREG2
FREQ <sample_rate>
GAIN ADS1 <channel> <gain>
GAIN ADS2 <channel> <gain>
INPUT ADS1 <channel> <mode>
INPUT ADS2 <channel> <mode>
SRB1ON: ADS1
SRB1OFF: ADS1
SRB1ON: ADS2
SRB1OFF: ADS2
ADS2ON
ADSOFF
```

Expected data/register lines:

```text
Ch1: v1,v2,v3,v4,v5,v6,v7,v8
Ch2: v1,v2,...,v16
ADS1: REG[0x05] = 0x96
ADS2: REG[0x05] = 0x96
Status: ...
```

The app does not currently write raw ADS1299 registers. Register fields are read-only displays updated from `READREG1` / `READREG2` responses.

