# istots deployment set

This folder contains the deployment-ready icon bundle derived from the retained SVG master.

## Source of truth

- SVG master:
  - `source/istots.svg`

## Outputs

- Generic PNG sizes:
  - `16, 20, 22, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 128, 256, 512, 1024`
- Windows:
  - `windows/istots.ico`
  - `windows/istots_setup.ico`
  - `windows/app_list_targetsize/`
- macOS:
  - `macos/istots.icns`
  - `macos/istots.iconset/`
- Linux:
  - `linux/hicolor/scalable/apps/istots.svg`
  - `linux/hicolor/<size>x<size>/apps/istots.png`
- Inno Setup snippet:
  - `inno_setup/inno_setup_snippet.iss`

## Notes

- `windows/istots_setup.ico` is intended for Inno Setup `SetupIconFile`.
- `windows/istots.ico` can be bundled with the installed app and referenced by shortcuts.
- `macos/istots.icns` and `macos/istots.iconset/` are both included.
- All assets in this folder were generated from the retained master SVG only.
