# FaceTracker

Cross-platform neural face tracking client for the FaceTrack Minecraft mod.

The implementation, build instructions, and native launchers are in
[`main_client`](main_client/README.md). The client supports Linux, Windows, and
macOS.

Quick validation:

```bash
./main_client/run.sh --check-models
```

Run on Linux or macOS:

```bash
./main_client/run.sh --camera 0 --mirror
```

Run on Windows:

```powershell
.\main_client\run.ps1 --camera 0 --mirror
```
