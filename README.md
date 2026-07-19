# FaceTracker

Cross-platform neural face tracking client for the FaceTrack Minecraft mod.

The implementation, build instructions, and native launchers are in
[`linux_client2`](linux_client2/README.md). Despite the directory's legacy name,
the client supports Linux, Windows, and macOS.

Quick validation:

```bash
./linux_client2/run.sh --check-models
```

Run on Linux or macOS:

```bash
./linux_client2/run.sh --camera 0 --mirror
```

Run on Windows:

```powershell
.\linux_client2\run.ps1 --camera 0 --mirror
```
