# kyty-gpu-hybrid

A fork of KytyPS5 by [TheCruZ](https://github.com/TheCruZ), itself based on the original
[Kyty](https://github.com/InoriRus/Kyty) emulator by InoriRus.

This fork builds Kyty's GPU (the guest command processor, the shader recompiler and the Vulkan
renderer) so that another PS5 compatibility layer can load it as a DLL and use it as its graphics
backend. The rest of Kyty is kept so the fork can still be built and tested on its own.

Branches:
- `hibrido`: the GPU DLL build and the fixes it needs.
- `kyty-puro`: the AMD fixes only, on top of KytyPS5, without the DLL changes.

## License

GPL-2.0, as KytyPS5 (see `LICENSE`). The original Kyty code is MIT (see `LICENSES/Kyty-MIT.txt`).

## Credits

- InoriRus, for Kyty.
- TheCruZ and the KytyPS5 contributors, for the fork this one starts from.

Test suite: 37/79 pass on Windows/clang-cl; the failures predate this fork's changes.

No game files, keys or firmware are part of this repository.
